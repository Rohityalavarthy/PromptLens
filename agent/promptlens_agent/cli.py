import asyncio
import click
from pathlib import Path
from .discovery import discover_prompts
from .reporter import print_saliency_report, print_audit_summary, print_compression_result
from .compressor import compress_prompt
from .validator import validate_compression
from promptlens import run_shapley, SimilarityMode, CompressionResult


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
def compress(file: str, threshold: float, semantic: bool, test_inputs_file: str | None, m_samples: int):
    """
    Full compression pipeline: analyse → rewrite → validate → write .suggested file.
    Does not overwrite the original. Developer reviews and accepts manually.
    """
    prompt_path = Path(file)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(test_inputs_file)
    mode = _get_mode(semantic)

    async def run():
        # Step 1: Shapley analysis
        click.echo(f"⚙  Running Shapley analysis (M={m_samples})...")
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=m_samples,
            mode=mode,
        )
        print_saliency_report(report, file=file)

        # Step 2: Constrained compression
        click.echo("✂  Compressing low-saliency phrases...")
        compressed, diff = await compress_prompt(report)

        # Step 3: Validation loop
        click.echo(f"🔍 Validating against {len(test_inputs)} test inputs (threshold={threshold})...")
        passed, worst_divergence, final_compressed = await validate_compression(
            original_prompt=prompt_text,
            compressed_prompt=compressed,
            test_inputs=test_inputs,
            report=report,
            diff=diff,
            threshold=threshold,
            mode=mode,
        )

        # Step 4: Write output
        original_tokens = len(prompt_text.split())
        compressed_tokens = len(final_compressed.split())
        result = CompressionResult(
            original_prompt=prompt_text,
            compressed_prompt=final_compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_delta=original_tokens - compressed_tokens,
            validation_passed=passed,
            worst_case_divergence=worst_divergence,
            saliency_report=report,
            diff=diff,
        )

        suggested_path = prompt_path.with_suffix(prompt_path.suffix + ".suggested")
        suggested_path.write_text(final_compressed, encoding="utf-8")

        print_compression_result(result)

        if not passed:
            click.echo("⚠️  Validation did not fully pass. Review .suggested file carefully before adopting.")

    asyncio.run(run())


def main():
    cli()
