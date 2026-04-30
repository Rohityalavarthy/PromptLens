import asyncio
import re
import click
from pathlib import Path
from .discovery import discover_prompts, discover_prompts_in_file
from .reporter import print_saliency_report, print_audit_summary, print_compression_result, make_progress_callback
from .compressor import compress_prompt
from .validator import validate_compression
from promptlens import run_shapley, SimilarityMode, CompressionResult


def _resolve_compress_target(
    prompt_path: Path,
    raw_source: str,
) -> tuple[str, Path, callable]:
    """
    Determine what content to compress and how to reconstruct the output file.

    Returns (prompt_text, write_path, make_output) where:
      prompt_text  — the string to run Shapley analysis and compression on
      write_path   — Path to write the result to (may differ from prompt_path
                     when the prompt lives in a separate file)
      make_output  — callable(compressed: str) -> str: builds the full file
                     content to write, splicing compressed back into the
                     original structure

    Resolution order for .py files:
      1. AST-based discovery (handles API call patterns, variables, file refs)
      2. Regex fallback for standalone prompt strings without an API call
      3. Whole-file fallback
    """
    if prompt_path.suffix == ".py":
        # --- Primary: AST discovery ---
        findings = discover_prompts_in_file(str(prompt_path))
        hit = next((d for d in findings if d.prompt_text), None)
        if hit:
            if hit.origin == "file" and hit.origin_file:
                # Prompt lives in an external file — write there directly
                ext = Path(hit.origin_file)
                if not ext.is_absolute():
                    ext = prompt_path.parent / ext
                return hit.prompt_text, ext, lambda c: c

            # Inline literal or variable: replace the prompt value in the source.
            # str.replace(..., 1) is safe because the AST extracted this exact
            # string from the same source, so it must be present.
            original = hit.prompt_text
            return original, prompt_path, lambda c, _o=original, _s=raw_source: _s.replace(_o, c, 1)

        # --- Fallback: regex for prompt-only files with no API call ---
        for delim in ('"""', "'''"):
            m = re.search(re.escape(delim) + r"(.*?)" + re.escape(delim), raw_source, re.DOTALL)
            if m:
                prefix = raw_source[:m.start(1)]
                content = m.group(1)
                suffix = raw_source[m.end(1):]
                return content, prompt_path, lambda c, _p=prefix, _s=suffix: _p + c + _s

    # --- Non-Python or no match: whole file is the prompt ---
    return raw_source, prompt_path, lambda c: c


def _get_mode(semantic: bool) -> SimilarityMode:
    return SimilarityMode.SEMANTIC if semantic else SimilarityMode.STANDARD


def _load_test_inputs(test_inputs_file: str | None, n: int = 10) -> list[str]:
    """Load test inputs from a .jsonl or .txt file, or return a minimal fallback set."""
    if test_inputs_file:
        path = Path(test_inputs_file)
        if path.suffix == ".jsonl":
            import json
            lines = path.read_text().strip().splitlines()
            return [json.loads(l)["input"] for l in lines[:n]]
        else:
            return [l.strip() for l in path.read_text().splitlines() if l.strip()][:n]

    # Fallback: three generic inputs that exercise most system prompts reasonably well
    return [
        "Please help me with this task.",
        "What should I do in this situation?",
        "Give me your best recommendation.",
    ]


@click.group()
def cli():
    """PromptLens Agent — Evidence-based prompt optimisation."""
    pass


@cli.command()
@click.option("--file", "-f", required=True, help="Path to prompt file to check.")
@click.option("--semantic", is_flag=True, default=False, help="Use semantic similarity (requires Together AI key).")
def check(file: str, semantic: bool):
    """
    Fast saliency check on a single prompt file. M=3 samples.
    Use before commits to catch newly introduced bloat.
    """
    prompt_text = Path(file).read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(None, n=3)

    async def run():
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=3,
            mode=_get_mode(semantic),
            on_progress=make_progress_callback("Shapley"),
        )
        print_saliency_report(report, file=file)
        if report.redundancy_fraction > 0.2:
            click.echo(f"⚠️  {report.redundancy_fraction*100:.0f}% of phrases appear low-impact.")
            click.echo(f"   Run: promptlens compress --file {file}")

    asyncio.run(run())


@cli.command()
@click.option("--repo", "-r", default=".", help="Path to repository root. Default: current directory.")
@click.option("--file", "-f", default=None, help="Analyse a specific file only.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None, help="Path to .jsonl or .txt file of test inputs.")
@click.option("--m-samples", default=20, help="Monte Carlo samples per test input. Default: 20.")
def audit(repo: str, file: str | None, semantic: bool, test_inputs_file: str | None, m_samples: int):
    """
    Full saliency audit. Discovers all prompts in repo (or analyses a single file).
    Outputs a compression brief per prompt.
    """
    if file:
        targets = [Path(file)]
    else:
        discoveries = discover_prompts(repo)
        targets = [
            Path(d.origin_file or d.file)
            for d in discoveries
            if d.prompt_text
        ]
        if not targets:
            click.echo("No prompts found. Is this a Python codebase with OpenAI/Anthropic calls?")
            return

    async def run():
        reports = {}
        for target in targets:
            prompt_text = target.read_text(encoding="utf-8")
            test_inputs = _load_test_inputs(test_inputs_file)
            report = await run_shapley(
                prompt=prompt_text,
                test_inputs=test_inputs,
                m_samples=m_samples,
                mode=_get_mode(semantic),
                on_progress=make_progress_callback("Shapley"),
            )
            reports[str(target)] = report
            print_saliency_report(report, file=str(target))

        if len(reports) > 1:
            print_audit_summary(targets, reports)

    asyncio.run(run())


@cli.command()
@click.option("--file", "-f", required=True, help="Path to prompt file to compress.")
@click.option("--threshold", default=0.15, help="Max output divergence allowed. Default: 0.15.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None)
@click.option("--m-samples", default=20)
@click.option("--apply", "apply_changes", is_flag=True, default=False, help="Prompt to apply changes in-place after compression.")
def compress(file: str, threshold: float, semantic: bool, test_inputs_file: str | None, m_samples: int, apply_changes: bool):
    """
    Full compression pipeline: analyse → rewrite → validate → write .suggested file.
    Does not overwrite the original. Developer reviews and accepts manually.
    """
    prompt_path = Path(file)
    raw_source = prompt_path.read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(test_inputs_file)
    mode = _get_mode(semantic)

    prompt_text, write_path, make_output = _resolve_compress_target(prompt_path, raw_source)

    async def run():
        # Step 1: Shapley analysis
        click.echo(f"⚙  Running Shapley analysis (M={m_samples})...")
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=m_samples,
            mode=mode,
            on_progress=make_progress_callback("Shapley"),
        )
        print_saliency_report(report, file=file)

        # Step 2: Constrained compression — LLM sees scores + threshold
        click.echo("✂  Compressing low-saliency phrases...")
        compressed, diff = await compress_prompt(report, threshold=threshold)

        # Step 3: Validation loop
        click.echo(f"🔍 Validating against {len(test_inputs)} test inputs (threshold={threshold})...")
        verdict, worst_divergence, final_compressed = await validate_compression(
            original_prompt=prompt_text,
            compressed_prompt=compressed,
            test_inputs=test_inputs,
            report=report,
            diff=diff,
            threshold=threshold,
            mode=mode,
            on_progress=make_progress_callback("Validation"),
        )

        # Step 4: Reconstruct full file, preserving surrounding code structure
        final_output = make_output(final_compressed)

        original_tokens = len(prompt_text.split())
        compressed_tokens = len(final_compressed.split())
        result = CompressionResult(
            original_prompt=prompt_text,
            compressed_prompt=final_compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_delta=original_tokens - compressed_tokens,
            validation_verdict=verdict,
            worst_case_divergence=worst_divergence,
            saliency_report=report,
            diff=diff,
        )

        suggested_path = write_path.with_suffix(write_path.suffix + ".suggested")
        suggested_path.write_text(final_output, encoding="utf-8")

        print_compression_result(result)

        if apply_changes:
            click.echo()
            if verdict == "FAIL":
                click.echo("Validation failed — applying is not recommended.")
            elif verdict in ("MARGINAL", "REVIEW"):
                click.echo("Divergence detected — review .suggested before applying.")
            confirmed = click.confirm("Apply compressed version? This will overwrite the original.", default=False)
            if confirmed:
                write_path.write_text(final_output, encoding="utf-8")
                suggested_path.unlink(missing_ok=True)
                click.echo(f"✓ Applied. {write_path} updated.")
            else:
                click.echo(f"Not applied. Suggested version is at: {suggested_path}")

    asyncio.run(run())


def main():
    cli()
