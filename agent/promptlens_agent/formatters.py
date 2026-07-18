"""Structured output formatters for PromptLens CLI.

Provides JSON and SARIF v2.1.0 output for all CLI commands, enabling
CI/CD integration, programmatic consumption, and IDE tooling.
"""

import json
from typing import Any


def format_saliency_json(scores: list, token_count: int, file: str, threshold: float) -> str:
    """JSON output for saliency check."""
    redundancy_fraction = sum(1 for s in scores if s.score < threshold) / len(scores) if scores else 0
    compression_candidates = sum(1 for s in scores if s.score < threshold)

    return json.dumps({
        "version": "1.0",
        "command": "check",
        "file": file,
        "token_count": token_count,
        "phrase_count": len(scores),
        "redundancy_fraction": round(redundancy_fraction, 4),
        "compression_candidate_count": compression_candidates,
        "threshold": threshold,
        "phrases": [
            {
                "index": s.phrase.index,
                "text": s.phrase.text,
                "score": round(s.score, 4),
                "raw_shapley": round(s.raw_shapley, 4) if hasattr(s, 'raw_shapley') else round(s.score, 4),
                "disposition": "remove" if s.score < threshold * 0.5 else ("compress" if s.score < threshold else "keep"),
                "char_start": s.phrase.char_start,
                "char_end": s.phrase.char_end,
                "region_type": s.phrase.region_type.value if hasattr(s.phrase.region_type, 'value') else str(s.phrase.region_type),
                "atomic": s.phrase.atomic,
            }
            for s in scores
        ],
    }, indent=2)


def format_audit_json(results: dict[str, dict]) -> str:
    """JSON output for audit — multiple files."""
    return json.dumps({
        "version": "1.0",
        "command": "audit",
        "files": results,
    }, indent=2)


def format_compression_json(
    compressed_prompt: str,
    diff: list[dict],
    scores: list,
    original_prompt: str,
    verdict: str,
    worst_divergence: float,
    threshold: float,
) -> str:
    """JSON output for compression with full explainability."""
    original_tokens = len(original_prompt.split())
    compressed_tokens = len(compressed_prompt.split())

    return json.dumps({
        "version": "1.0",
        "command": "compress",
        "validation_verdict": verdict,
        "worst_case_divergence": round(worst_divergence, 4),
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "token_reduction_pct": round((1 - compressed_tokens / original_tokens) * 100, 1) if original_tokens > 0 else 0,
        "threshold": threshold,
        "compressed_prompt": compressed_prompt,
        "decisions": [
            {
                "phrase_index": entry.get("phrase", i),
                "action": entry["action"],
                "original": entry.get("original", ""),
                "result": entry.get("result", ""),
                "merge_target": entry.get("merge_target"),
                "saliency_score": round(scores[entry.get("phrase", i)].score, 4) if entry.get("phrase", i) < len(scores) else None,
                "rationale": _action_rationale(entry, scores),
            }
            for i, entry in enumerate(diff)
        ],
        "validation": {
            "verdict": verdict,
            "worst_divergence": round(worst_divergence, 4),
            "threshold": threshold,
        },
    }, indent=2)


def _action_rationale(entry: dict, scores: list) -> str:
    """Generate deterministic rationale for compression decision."""
    phrase_idx = entry.get("phrase", 0)
    if phrase_idx < len(scores):
        score = scores[phrase_idx].score
    else:
        score = 1.0

    action = entry["action"]
    if action == "keep":
        return f"Retained: saliency score {score:.2f} indicates meaningful contribution."
    elif action == "remove":
        return f"Removed: saliency score {score:.2f} indicates negligible contribution to output quality."
    elif action == "rewrite":
        return f"Rewritten: saliency score {score:.2f} suggests content is partially redundant; compressed while preserving semantic intent."
    elif action == "merge":
        target = entry.get("merge_target", "?")
        return f"Merged into phrase #{target}: content overlaps with target phrase; combined for conciseness."
    elif action == "paraphrase":
        return f"Paraphrased: saliency score {score:.2f} allows shorter expression without semantic loss."
    return f"Action '{action}' applied based on saliency score {score:.2f}."


def _build_sarif_result(score, file: str, threshold: float) -> dict:
    """Build a single SARIF result entry for a low-saliency phrase."""
    level = "warning" if score.score < 0.10 else "note"
    disposition = "remove" if score.score < threshold * 0.5 else "compress"
    return {
        "ruleId": "promptlens/low-saliency",
        "level": level,
        "message": {"text": f"Phrase scores {score.score:.3f} (below threshold {threshold}). Disposition: {disposition}."},
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": file},
                "region": {
                    "charOffset": score.phrase.char_start if score.phrase.char_start >= 0 else 0,
                    "charLength": (score.phrase.char_end - score.phrase.char_start) if score.phrase.char_start >= 0 else len(score.phrase.text),
                }
            }
        }],
    }


def _build_sarif_envelope(results: list[dict]) -> dict:
    """Build the SARIF v2.1.0 envelope around results."""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "PromptLens",
                    "version": "0.1.0",
                    "informationUri": "https://github.com/Rohityalavarthy/PromptLens",
                    "rules": [{
                        "id": "promptlens/low-saliency",
                        "shortDescription": {"text": "Low-saliency phrase detected"},
                    }]
                }
            },
            "results": results,
        }]
    }


def format_saliency_sarif(scores: list, token_count: int, file: str, threshold: float) -> str:
    """SARIF v2.1.0 output for saliency findings."""
    results = []
    for s in scores:
        if s.score >= threshold:
            continue  # Only report low-saliency phrases
        result = _build_sarif_result(s, file, threshold)
        # Add text snippet to message for single-file check
        text = s.phrase.text
        if len(text) > 80:
            result["message"]["text"] += f" Text: \"{text[:80]}...\""
        else:
            result["message"]["text"] += f" Text: \"{text}\""
        results.append(result)

    return json.dumps(_build_sarif_envelope(results), indent=2)


def format_audit_sarif(all_scores: dict[str, list], threshold: float) -> str:
    """SARIF output for full audit — one run, results from all files."""
    results = []
    for file, scores in all_scores.items():
        for s in scores:
            if s.score >= threshold:
                continue
            results.append(_build_sarif_result(s, file, threshold))

    return json.dumps(_build_sarif_envelope(results), indent=2)
