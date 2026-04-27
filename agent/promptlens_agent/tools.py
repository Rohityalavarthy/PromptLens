"""
LangGraph-compatible tool definitions for PromptLens Agent.

These wrap the core operations as @tool-decorated functions so they can be
composed in a LangGraph StateGraph for future smart-mode orchestration — e.g.,
an agent that decides which prompts to audit, whether to compress, and how to
prioritise based on redundancy scores.
"""
import asyncio
from langchain_core.tools import tool
from promptlens import run_shapley, SimilarityMode
from .discovery import discover_prompts, DiscoveredPrompt
from .compressor import compress_prompt
from .validator import validate_compression


@tool
def scan_codebase(repo_path: str) -> list[dict]:
    """
    Discover all LLM prompts in a Python codebase.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        List of dicts with keys: file, line, framework, origin, estimated_tokens,
        and prompt_text (None if origin is 'unknown' or 'variable').
    """
    discoveries = discover_prompts(repo_path)
    return [
        {
            "file": d.file,
            "line": d.line,
            "framework": d.framework,
            "origin": d.origin,
            "estimated_tokens": d.estimated_tokens,
            "prompt_text": d.prompt_text,
        }
        for d in discoveries
    ]


@tool
def analyze_prompt(
    prompt: str,
    test_inputs: list[str],
    m_samples: int = 20,
    use_semantic: bool = False,
) -> dict:
    """
    Run Shapley saliency analysis on a system prompt.

    Args:
        prompt: The system prompt text to analyse.
        test_inputs: Representative user messages to test against.
        m_samples: Monte Carlo walks per test input (3=fast, 20=balanced, 50=precise).
        use_semantic: If True, uses embedding cosine similarity (requires Together AI key).

    Returns:
        Dict with keys: token_count, redundancy_fraction, confidence,
        compression_candidate_tokens, and per_phrase scores (list of {text, score, disposition}).
    """
    mode = SimilarityMode.SEMANTIC if use_semantic else SimilarityMode.STANDARD

    async def _run():
        return await run_shapley(prompt=prompt, test_inputs=test_inputs, m_samples=m_samples, mode=mode)

    report = asyncio.run(_run())

    return {
        "token_count": report.token_count,
        "redundancy_fraction": report.redundancy_fraction,
        "confidence": report.confidence,
        "compression_candidate_tokens": report.compression_candidate_tokens,
        "per_phrase": [
            {
                "text": s.phrase.text,
                "score": s.score,
                "disposition": s.disposition,
            }
            for s in report.scores
        ],
    }


@tool
def compress_and_validate(
    prompt: str,
    test_inputs: list[str],
    threshold: float = 0.15,
    m_samples: int = 20,
    use_semantic: bool = False,
) -> dict:
    """
    Full compression pipeline: analyse → rewrite low-saliency phrases → validate.

    Args:
        prompt: The system prompt text to compress.
        test_inputs: Representative user messages used for validation.
        threshold: Max output divergence allowed (0.0–1.0). Default: 0.15.
        m_samples: Monte Carlo walks for Shapley analysis.
        use_semantic: Use semantic similarity mode.

    Returns:
        Dict with keys: compressed_prompt, original_tokens, compressed_tokens,
        token_delta, validation_passed, worst_case_divergence.
    """
    mode = SimilarityMode.SEMANTIC if use_semantic else SimilarityMode.STANDARD

    async def _run():
        report = await run_shapley(prompt=prompt, test_inputs=test_inputs, m_samples=m_samples, mode=mode)
        compressed, diff = await compress_prompt(report)
        passed, worst_div, final = await validate_compression(
            original_prompt=prompt,
            compressed_prompt=compressed,
            test_inputs=test_inputs,
            report=report,
            diff=diff,
            threshold=threshold,
            mode=mode,
        )
        return final, len(prompt.split()), len(final.split()), passed, worst_div

    final, orig_tokens, comp_tokens, passed, worst_div = asyncio.run(_run())

    return {
        "compressed_prompt": final,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "token_delta": orig_tokens - comp_tokens,
        "validation_passed": passed,
        "worst_case_divergence": worst_div,
    }
