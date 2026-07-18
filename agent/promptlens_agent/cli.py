import asyncio
import json
import sys
import re
import click
from pathlib import Path
from .discovery import discover_prompts, discover_prompts_in_file
from .reporter import print_saliency_report, print_audit_summary, print_compression_result, make_progress_callback
from .compressor import compress_prompt
from .validator import validate_compression
from .config import load_config, PromptLensConfig
from .formatters import (
    format_saliency_json,
    format_saliency_sarif,
    format_audit_json,
    format_audit_sarif,
    format_compression_json,
)
from .cost_estimator import (
    estimate_shapley_cost,
    estimate_compression_cost,
    format_estimate,
    estimate_to_dict,
)
from promptlens import run_shapley, segment_prompt, SimilarityMode, CompressionResult


def _resolve_option(cli_value, config_value):
    """CLI flag wins if explicitly set (not None), else use config value."""
    return cli_value if cli_value is not None else config_value


def _setup_provider_from_config(config):
    """Configure the LLM provider based on config settings."""
    import os
    from promptlens.provider import TogetherProvider, OpenAIProvider, AnthropicProvider
    from promptlens.generator import configure_provider

    PROVIDER_MAP = {"together": TogetherProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}
    provider_cls = PROVIDER_MAP.get(config.provider, TogetherProvider)

    kwargs = {}
    if config.model:
        kwargs["generator_model"] = config.model

    # Determine API key
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
    else:
        default_env = {"together": "TOGETHER_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        api_key = os.environ.get(default_env.get(config.provider, "TOGETHER_API_KEY"))

    if api_key:
        kwargs["api_key"] = api_key

    provider = provider_cls(**kwargs)
    configure_provider(provider)


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
            import json as json_mod
            lines = path.read_text().strip().splitlines()
            return [json_mod.loads(l)["input"] for l in lines[:n]]
        else:
            return [l.strip() for l in path.read_text().splitlines() if l.strip()][:n]

    # Fallback: three generic inputs that exercise most system prompts reasonably well
    return [
        "Please help me with this task.",
        "What should I do in this situation?",
        "Give me your best recommendation.",
    ]


def _echo(msg: str, output_format: str, **kwargs):
    """Print a message: to stderr if structured output active, else normal echo."""
    if output_format != "terminal":
        click.echo(msg, err=True, **kwargs)
    else:
        click.echo(msg, **kwargs)


@click.group()
def cli():
    """PromptLens Agent — Evidence-based prompt optimisation."""
    pass


@cli.command()
@click.option("--file", "-f", required=True, help="Path to prompt file to check.")
@click.option("--semantic", is_flag=True, default=False, help="Use semantic similarity (requires Together AI key).")
@click.option("--saliency-threshold", "saliency_threshold", default=None, type=float, help="Phrases scoring below this are flagged as low-impact. Default: 0.15.")
@click.option("--format", "output_format", type=click.Choice(["terminal", "json", "sarif"]), default=None,
              help="Output format (default: terminal, or from config)")
@click.option("--ci", is_flag=True, default=False, help="Non-interactive CI mode.")
def check(file: str, semantic: bool, saliency_threshold: float | None, output_format: str | None, ci: bool):
    """
    Fast saliency check on a single prompt file. M=3 samples.
    Use before commits to catch newly introduced bloat.
    """
    try:
        config = load_config()
        _setup_provider_from_config(config)
        if not semantic:
            semantic = config.semantic
        saliency_threshold = _resolve_option(saliency_threshold, config.saliency_threshold)
        # CI mode defaults to JSON if no explicit --format was given
        if ci and output_format is None:
            output_format = "json"
        output_format = _resolve_option(output_format, config.output_format)
        if output_format is None:
            output_format = "terminal"

        prompt_text = Path(file).read_text(encoding="utf-8")
        test_inputs = _load_test_inputs(None, n=3)

        async def run():
            report = await run_shapley(
                prompt=prompt_text,
                test_inputs=test_inputs,
                m_samples=3,
                mode=_get_mode(semantic),
                low_saliency_threshold=saliency_threshold,
                on_progress=make_progress_callback("Shapley"),
            )

            if output_format == "json":
                result = format_saliency_json(report.scores, report.token_count, file, saliency_threshold)
                click.echo(result, file=sys.stdout)
            elif output_format == "sarif":
                result = format_saliency_sarif(report.scores, report.token_count, file, saliency_threshold)
                click.echo(result, file=sys.stdout)
            else:
                print_saliency_report(report, file=file)
                if report.redundancy_fraction > 0.2:
                    click.echo(f"\u26a0\ufe0f  {report.redundancy_fraction*100:.0f}% of phrases appear low-impact.")
                    click.echo(f"   Run: promptlens compress --file {file}")

            # Exit code: 1 if redundancy > 20%
            if report.redundancy_fraction > 0.2:
                sys.exit(1)

        asyncio.run(run())
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)


@cli.command()
@click.option("--repo", "-r", default=".", help="Path to repository root. Default: current directory.")
@click.option("--file", "-f", default=None, help="Analyse a specific file only.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None, help="Path to .jsonl or .txt file of test inputs.")
@click.option("--m-samples", default=None, type=int, help="Monte Carlo samples per test input. Default: 20.")
@click.option("--saliency-threshold", "saliency_threshold", default=None, type=float, help="Phrases scoring below this are flagged as low-impact. Default: 0.15.")
@click.option("--format", "output_format", type=click.Choice(["terminal", "json", "sarif"]), default=None,
              help="Output format (default: terminal, or from config)")
@click.option("--ci", is_flag=True, default=False, help="Non-interactive CI mode.")
def audit(repo: str, file: str | None, semantic: bool, test_inputs_file: str | None, m_samples: int | None, saliency_threshold: float | None, output_format: str | None, ci: bool):
    """
    Full saliency audit. Discovers all prompts in repo (or analyses a single file).
    Outputs a compression brief per prompt.
    """
    try:
        config = load_config()
        _setup_provider_from_config(config)
        if not semantic:
            semantic = config.semantic
        m_samples = _resolve_option(m_samples, config.m_samples)
        saliency_threshold = _resolve_option(saliency_threshold, config.saliency_threshold)
        test_inputs_file = _resolve_option(test_inputs_file, config.test_inputs_file)
        # CI mode defaults to JSON if no explicit --format was given
        if ci and output_format is None:
            output_format = "json"
        output_format = _resolve_option(output_format, config.output_format)
        if output_format is None:
            output_format = "terminal"

        if file:
            targets = [(Path(file), None)]  # (path, prompt_text) — None means read from file
        else:
            discoveries = discover_prompts(repo)
            targets = [
                (Path(d.origin_file or d.file), d.prompt_text)
                for d in discoveries
                if d.prompt_text
            ]
            if not targets:
                _echo("No prompts found. Is this a Python codebase with OpenAI/Anthropic calls?", output_format)
                return

        async def run():
            reports = {}
            all_scores = {}
            any_high_redundancy = False

            for target_path, discovered_text in targets:
                prompt_text = discovered_text if discovered_text is not None else target_path.read_text(encoding="utf-8")
                test_inputs = _load_test_inputs(test_inputs_file)
                report = await run_shapley(
                    prompt=prompt_text,
                    test_inputs=test_inputs,
                    m_samples=m_samples,
                    mode=_get_mode(semantic),
                    low_saliency_threshold=saliency_threshold,
                    on_progress=make_progress_callback("Shapley"),
                )
                reports[str(target_path)] = report
                all_scores[str(target_path)] = report.scores

                if report.redundancy_fraction > 0.2:
                    any_high_redundancy = True

                if output_format == "terminal":
                    print_saliency_report(report, file=str(target_path))

            if output_format == "json":
                # Build per-file summary for JSON
                json_results = {}
                for fname, report in reports.items():
                    json_results[fname] = {
                        "token_count": report.token_count,
                        "phrase_count": len(report.scores),
                        "redundancy_fraction": round(report.redundancy_fraction, 4),
                    }
                result = format_audit_json(json_results)
                click.echo(result, file=sys.stdout)
            elif output_format == "sarif":
                result = format_audit_sarif(all_scores, saliency_threshold)
                click.echo(result, file=sys.stdout)
            else:
                if len(reports) > 1:
                    print_audit_summary([t[0] for t in targets], reports)

            # Exit code: 1 if any file has redundancy > 20%
            if any_high_redundancy:
                sys.exit(1)

        asyncio.run(run())
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)


@cli.command()
@click.option("--file", "-f", default=None, help="Path to prompt file to compress.")
@click.option("--threshold", default=None, type=float, help="Max output divergence allowed by validation. Default: 0.15.")
@click.option("--saliency-threshold", "saliency_threshold", default=None, type=float,
              help="Phrases scoring below this are flagged for compression. "
                   "Defaults to min(--threshold, 0.50) — scales automatically with --threshold.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None)
@click.option("--m-samples", default=None, type=int)
@click.option("--apply", "apply_changes", is_flag=True, default=False, help="Prompt to apply changes in-place after compression.")
@click.option("--format", "output_format", type=click.Choice(["terminal", "json", "sarif"]), default=None,
              help="Output format (default: terminal, or from config)")
@click.option("--dry-run", is_flag=True, default=False, help="Estimate cost without running. No API calls made.")
@click.option("--batch", is_flag=True, default=False, help="Compress all discovered prompts in repo.")
@click.option("--repo", "-r", default=".", help="Path to repository root for batch mode. Default: current directory.")
@click.option("--ci", is_flag=True, default=False, help="Non-interactive CI mode.")
def compress(file: str | None, threshold: float | None, saliency_threshold: float | None, semantic: bool, test_inputs_file: str | None, m_samples: int | None, apply_changes: bool, output_format: str | None, dry_run: bool, batch: bool, repo: str, ci: bool):
    """
    Full compression pipeline: analyse → rewrite → validate → write .suggested file.
    Does not overwrite the original. Developer reviews and accepts manually.
    """
    try:
        # Validate mutually exclusive options
        if batch and file:
            click.echo("Error: Cannot use --batch and --file together", err=True)
            sys.exit(2)
        if not batch and not file:
            click.echo("Error: Either --file or --batch is required", err=True)
            sys.exit(2)

        config = load_config()
        _setup_provider_from_config(config)
        if not semantic:
            semantic = config.semantic
        threshold = _resolve_option(threshold, config.threshold)
        m_samples = _resolve_option(m_samples, config.m_samples)
        test_inputs_file = _resolve_option(test_inputs_file, config.test_inputs_file)
        # CI mode defaults to JSON if no explicit --format was given
        if ci and output_format is None:
            output_format = "json"
        output_format = _resolve_option(output_format, config.output_format)
        if output_format is None:
            output_format = "terminal"

        mode = _get_mode(semantic)

        # Auto-derive saliency threshold: scales with --threshold so raising the
        # validation gate also flags more phrases for compression.
        effective_saliency = saliency_threshold if saliency_threshold is not None else min(threshold, 0.50)

        if batch:
            _run_batch_compress(
                repo=repo,
                threshold=threshold,
                effective_saliency=effective_saliency,
                semantic=semantic,
                test_inputs_file=test_inputs_file,
                m_samples=m_samples,
                output_format=output_format,
                dry_run=dry_run,
                ci=ci,
                mode=mode,
            )
        else:
            _run_single_compress(
                file=file,
                threshold=threshold,
                effective_saliency=effective_saliency,
                semantic=semantic,
                test_inputs_file=test_inputs_file,
                m_samples=m_samples,
                apply_changes=apply_changes,
                output_format=output_format,
                dry_run=dry_run,
                ci=ci,
                mode=mode,
            )
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)


def _run_single_compress(
    file: str,
    threshold: float,
    effective_saliency: float,
    semantic: bool,
    test_inputs_file: str | None,
    m_samples: int,
    apply_changes: bool,
    output_format: str,
    dry_run: bool,
    ci: bool,
    mode: SimilarityMode,
):
    """Run compress on a single file."""
    prompt_path = Path(file)
    raw_source = prompt_path.read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(test_inputs_file)

    prompt_text, write_path, make_output = _resolve_compress_target(prompt_path, raw_source)

    if dry_run:
        _run_dry_run(
            prompt_text=prompt_text,
            file=file,
            m_samples=m_samples,
            n_test_inputs=len(test_inputs),
            output_format=output_format,
        )
        return

    async def run():
        # Step 1: Shapley analysis
        _echo(f"\u2699  Running Shapley analysis (M={m_samples}, saliency-threshold={effective_saliency:.2f})...", output_format)
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=m_samples,
            mode=mode,
            low_saliency_threshold=effective_saliency,
            on_progress=make_progress_callback("Shapley"),
        )
        if output_format == "terminal":
            print_saliency_report(report, file=file)

        # Step 2: Constrained compression — LLM sees scores + threshold
        _echo("\u2702  Compressing low-saliency phrases...", output_format)
        compressed, diff = await compress_prompt(report, threshold=threshold)

        # Step 3: Validation loop
        _echo(f"\U0001f50d Validating against {len(test_inputs)} test inputs (threshold={threshold})...", output_format)
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

        # In CI mode, never write .suggested files
        if not ci:
            suggested_path = write_path.with_suffix(write_path.suffix + ".suggested")
            suggested_path.write_text(final_output, encoding="utf-8")

        if output_format == "json":
            json_output = format_compression_json(
                compressed_prompt=final_compressed,
                diff=diff,
                scores=report.scores,
                original_prompt=prompt_text,
                verdict=verdict,
                worst_divergence=worst_divergence,
                threshold=threshold,
            )
            click.echo(json_output, file=sys.stdout)
        elif output_format == "sarif":
            # For compress, SARIF reports the saliency findings
            sarif_output = format_saliency_sarif(report.scores, report.token_count, file, effective_saliency)
            click.echo(sarif_output, file=sys.stdout)
        else:
            print_compression_result(result)

        if output_format == "terminal" and apply_changes and not ci:
            click.echo()
            if verdict == "FAIL":
                click.echo("Validation failed — applying is not recommended.")
            elif verdict in ("MARGINAL", "REVIEW"):
                click.echo("Divergence detected — review .suggested before applying.")
            confirmed = click.confirm("Apply compressed version? This will overwrite the original.", default=False)
            if confirmed:
                write_path.write_text(final_output, encoding="utf-8")
                suggested_path.unlink(missing_ok=True)
                click.echo(f"\u2713 Applied. {write_path} updated.")
            else:
                click.echo(f"Not applied. Suggested version is at: {suggested_path}")

        # Exit code: 0 for PASS/MARGINAL, 1 for REVIEW/FAIL
        if verdict in ("REVIEW", "FAIL"):
            sys.exit(1)

    asyncio.run(run())


def _run_dry_run(
    prompt_text: str,
    file: str,
    m_samples: int,
    n_test_inputs: int,
    output_format: str,
):
    """Run dry-run mode: segment and estimate costs without API calls."""
    phrases = segment_prompt(prompt_text)
    n_phrases = len(phrases)
    token_count = len(prompt_text.split())

    shapley_est = estimate_shapley_cost(token_count, n_phrases, m_samples, n_test_inputs)
    compression_est = estimate_compression_cost(token_count, n_phrases, n_test_inputs)

    # Total estimate
    total_api_calls = shapley_est.api_calls + compression_est.api_calls
    total_input = shapley_est.estimated_input_tokens + compression_est.estimated_input_tokens
    total_output = shapley_est.estimated_output_tokens + compression_est.estimated_output_tokens
    total_cost = shapley_est.estimated_cost_usd + compression_est.estimated_cost_usd
    total_time = shapley_est.estimated_time_seconds + compression_est.estimated_time_seconds

    if output_format == "json":
        from .cost_estimator import CostEstimate
        total_est = CostEstimate(
            api_calls=total_api_calls,
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output,
            estimated_cost_usd=round(total_cost, 4),
            estimated_time_seconds=round(total_time, 1),
        )
        output = json.dumps({
            "version": "1.0",
            "command": "compress",
            "mode": "dry-run",
            "file": file,
            "token_count": token_count,
            "phrase_count": n_phrases,
            "shapley_estimate": estimate_to_dict(shapley_est),
            "compression_estimate": estimate_to_dict(compression_est),
            "total_estimate": estimate_to_dict(total_est),
        }, indent=2)
        click.echo(output, file=sys.stdout)
    else:
        click.echo(f"Dry-run estimate for: {file}")
        click.echo(f"  Token count: {token_count}")
        click.echo(f"  Phrase count: {n_phrases}")
        click.echo()
        click.echo("Shapley analysis:")
        click.echo(format_estimate(shapley_est))
        click.echo()
        click.echo("Compression + validation:")
        click.echo(format_estimate(compression_est))
        click.echo()
        click.echo(f"Total estimated cost: ${total_cost:.4f}")
        click.echo(f"Total estimated time: {total_time:.0f}s")


def _run_batch_compress(
    repo: str,
    threshold: float,
    effective_saliency: float,
    semantic: bool,
    test_inputs_file: str | None,
    m_samples: int,
    output_format: str,
    dry_run: bool,
    ci: bool,
    mode: SimilarityMode,
):
    """Run compress in batch mode across all discovered prompts."""
    discoveries = discover_prompts(repo)
    # Filter to prompts with confidence >= 0.5 and prompt_text is not None
    targets = [d for d in discoveries if d.confidence >= 0.5 and d.prompt_text is not None]

    if not targets:
        _echo("No prompts found for batch compression.", output_format)
        return

    test_inputs = _load_test_inputs(test_inputs_file)

    if dry_run:
        _run_batch_dry_run(targets, m_samples, len(test_inputs), output_format)
        return

    async def run():
        results = []
        any_failed = False

        for discovery in targets:
            file_path = discovery.origin_file or discovery.file
            prompt_text = discovery.prompt_text
            token_count = len(prompt_text.split())

            _echo(f"\nProcessing: {file_path}:{discovery.line}", output_format)

            try:
                # Step 1: Shapley analysis
                report = await run_shapley(
                    prompt=prompt_text,
                    test_inputs=test_inputs,
                    m_samples=m_samples,
                    mode=mode,
                    low_saliency_threshold=effective_saliency,
                    on_progress=make_progress_callback("Shapley"),
                )

                # Step 2: Compression
                compressed, diff = await compress_prompt(report, threshold=threshold)

                # Step 3: Validation
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

                original_tokens = len(prompt_text.split())
                compressed_tokens = len(final_compressed.split())
                reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0

                if verdict in ("REVIEW", "FAIL"):
                    any_failed = True

                results.append({
                    "file": file_path,
                    "line": discovery.line,
                    "verdict": verdict,
                    "reduction_pct": round(reduction, 1),
                    "worst_divergence": worst_divergence,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                })

            except Exception as exc:
                _echo(f"  Error processing {file_path}: {exc}", output_format)
                any_failed = True
                results.append({
                    "file": file_path,
                    "line": discovery.line,
                    "verdict": "FAIL",
                    "reduction_pct": 0,
                    "error": str(exc),
                })

        # Output results
        passed = sum(1 for r in results if r["verdict"] not in ("REVIEW", "FAIL"))
        failed = len(results) - passed

        if output_format == "json":
            output = json.dumps({
                "version": "1.0",
                "command": "compress",
                "mode": "batch",
                "results": results,
                "summary": {"total": len(results), "passed": passed, "failed": failed},
            }, indent=2)
            click.echo(output, file=sys.stdout)
        else:
            click.echo("\nBatch Results:")
            for r in results:
                status = r["verdict"]
                reduction = r.get("reduction_pct", 0)
                error = r.get("error")
                if error:
                    click.echo(f"  {r['file']}:{r['line']}  {status}   (error: {error})")
                else:
                    click.echo(f"  {r['file']}:{r['line']}  {status}   ({reduction:.0f}% reduction)")
            click.echo(f"\nSummary: {len(results)} files processed, {passed} passed, {failed} failed")

        if any_failed:
            sys.exit(1)

    asyncio.run(run())


def _run_batch_dry_run(targets, m_samples: int, n_test_inputs: int, output_format: str):
    """Run dry-run for batch mode."""
    results = []
    for discovery in targets:
        file_path = discovery.origin_file or discovery.file
        prompt_text = discovery.prompt_text
        phrases = segment_prompt(prompt_text)
        n_phrases = len(phrases)
        token_count = len(prompt_text.split())

        shapley_est = estimate_shapley_cost(token_count, n_phrases, m_samples, n_test_inputs)
        compression_est = estimate_compression_cost(token_count, n_phrases, n_test_inputs)

        total_cost = shapley_est.estimated_cost_usd + compression_est.estimated_cost_usd
        total_time = shapley_est.estimated_time_seconds + compression_est.estimated_time_seconds

        results.append({
            "file": file_path,
            "line": discovery.line,
            "token_count": token_count,
            "phrase_count": n_phrases,
            "shapley_estimate": estimate_to_dict(shapley_est),
            "compression_estimate": estimate_to_dict(compression_est),
            "total_cost_usd": round(total_cost, 4),
            "total_time_seconds": round(total_time, 1),
        })

    total_cost_all = sum(r["total_cost_usd"] for r in results)
    total_time_all = sum(r["total_time_seconds"] for r in results)

    if output_format == "json":
        output = json.dumps({
            "version": "1.0",
            "command": "compress",
            "mode": "dry-run",
            "batch": True,
            "results": results,
            "total_cost_usd": round(total_cost_all, 4),
            "total_time_seconds": round(total_time_all, 1),
        }, indent=2)
        click.echo(output, file=sys.stdout)
    else:
        click.echo("Batch Dry-Run Estimates:")
        for r in results:
            click.echo(f"  {r['file']}:{r['line']}  tokens={r['token_count']} phrases={r['phrase_count']} cost=${r['total_cost_usd']:.4f}")
        click.echo(f"\nTotal: {len(results)} files, estimated cost: ${total_cost_all:.4f}, time: {total_time_all:.0f}s")


def main():
    cli()
