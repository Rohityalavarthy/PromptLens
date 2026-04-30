import re
from promptlens import SaliencyReport, SaliencyScore
from promptlens.generator import generate


def reconstruct_from_original(original_prompt: str, diff: list[dict]) -> str:
    """
    Reconstruct a compressed prompt by substituting phrases directly in the
    original text. Preserves all whitespace, newlines, and structure.
    """
    result = original_prompt

    for entry in diff:
        if entry["action"] == "keep":
            continue
        old = entry["original"]
        new = entry["result"]  # "" for remove, replacement text for all others
        if old and old in result:
            result = result.replace(old, new, 1)

    # Clean up artifacts from removals
    result = re.sub(r'\n{3,}', '\n\n', result)       # multiple blank lines → one
    result = re.sub(r'\n[ \t]+\n', '\n\n', result)   # whitespace-only lines
    result = re.sub(r'[ \t]+\n', '\n', result)        # trailing spaces on lines

    return result.strip()


def _build_compression_prompt(threshold: float, redundancy_pct: int) -> str:
    """
    Single system prompt template. The threshold appears once as a concrete
    boundary in the score-based decision table — no discrete stance switching.
    redundancy_pct gives the LLM a concrete compression target derived from
    the Shapley analysis rather than a vague aggressiveness level.
    """
    # When threshold is at or below 0.10, the 0.10–threshold band is empty;
    # collapse it to keep the table clean.
    if threshold > 0.10:
        middle_row = (
            f"  score 0.10 – {threshold:.2f}:  "
            f"MERGE, REWRITE, or PARAPHRASE — do not REMOVE.\n"
        )
    else:
        middle_row = ""

    return f"""You are a prompt compression specialist. Reduce token count while \
preserving every distinct instruction and the full behavioural output of the prompt.

## Goal
Target approximately {redundancy_pct}% token reduction across the COMPRESS-labelled phrases.
All COMPRESS phrases have low Shapley saliency — they contribute little to model output \
when absent. KEEP phrases are high-impact — reproduce them EXACTLY, word for word.

## Available actions for COMPRESS phrases
  REMOVE      Phrase adds no unique instruction not covered by another phrase.
  MERGE       Phrase overlaps an adjacent COMPRESS phrase; combine into one concise line.
  REWRITE     Phrase is load-bearing but verbose; express the same instruction in fewer tokens.
  PARAPHRASE  Phrase can be restated more naturally or concisely while fully preserving \
meaning and tone.

## Score-based decision guide
  score 0.00 – 0.05:         REMOVE if covered elsewhere — near-zero impact, very safe.
  score 0.05 – 0.10:         REMOVE if fully covered; MERGE if adjacent overlap; \
else REWRITE or PARAPHRASE.
{middle_row}  score ≥ {threshold:.2f}:          PARAPHRASE only — borderline, quality gate is strict here.

## Output format — one line per input phrase, in the same order
  [KEEP] <exact original text>
  [REMOVE] <original text>
  [MERGE with <N>] <original text> → <merged result>
  [REWRITE] <original text> → <rewritten result>
  [PARAPHRASE] <original text> → <paraphrased result>

## Hard rules
  1. KEEP phrases must be reproduced EXACTLY — not one word changed.
  2. REWRITE, MERGE, and PARAPHRASE results must never be empty — use [REMOVE] if truly \
nothing unique remains.
  3. Every input phrase must appear exactly once in the output, in the same order.
  4. No explanations or commentary."""


async def compress_prompt(report: SaliencyReport, threshold: float = 0.15) -> tuple[str, list[dict]]:
    """
    Produce a compressed prompt from a SaliencyReport.
    Returns (compressed_prompt_text, diff_list).

    diff_list entries: {phrase, action, original, result}
    """
    redundancy_pct = (
        int(report.compression_candidate_tokens / report.token_count * 100)
        if report.token_count > 0 else 0
    )

    phrase_lines = []
    for score in report.scores:
        label = "KEEP" if score.disposition == "keep" else "COMPRESS"
        phrase_lines.append(f"[{label}:{score.score:.2f}] {score.phrase.text}")

    user_message = "Compress the following prompt phrases:\n\n" + "\n".join(phrase_lines)

    response = await generate(
        prompt=user_message,
        system_prompt=_build_compression_prompt(threshold, redundancy_pct),
        temperature=0.0,
        max_tokens=2048,
    )

    diff = _parse_compression_response(response, report.scores)
    compressed_prompt = reconstruct_from_original(report.prompt, diff)
    return compressed_prompt, diff


def _after_bracket(line: str) -> str:
    """Return the text that follows the first closing ] on a line."""
    close = line.find("]")
    return line[close + 1:].strip() if close != -1 else ""


def _result_after_arrow(text: str, fallback: str) -> str:
    """Extract the result half of 'original → result'. Returns fallback if no arrow."""
    arrow = text.find("→")
    return text[arrow + 1:].strip() if arrow != -1 else fallback


def _parse_compression_response(
    response: str,
    scores: list[SaliencyScore],
) -> list[dict]:
    """
    Parse the labelled compression output into a diff list.

    Robust to the LLM echoing scores in labels (e.g. [KEEP:0.94] instead of [KEEP])
    because text extraction uses the position of ] rather than fixed offsets.
    Falls back to keep-original for any unparseable line.
    """
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    diff = []

    for i, line in enumerate(lines):
        original = scores[i].phrase.text if i < len(scores) else ""

        if line.startswith("[KEEP"):
            text = _after_bracket(line)
            diff.append({"phrase": i, "action": "keep", "original": original, "result": text})

        elif line.startswith("[REMOVE"):
            diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})

        elif line.startswith("[MERGE"):
            result = _result_after_arrow(_after_bracket(line), fallback="")
            if not result:
                diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})
            else:
                diff.append({"phrase": i, "action": "merge", "original": original, "result": result})

        elif line.startswith("[REWRITE"):
            result = _result_after_arrow(_after_bracket(line), fallback="")
            if not result:
                diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})
            elif len(result.split()) >= len(original.split()):
                # LLM produced same length or longer — not a compression, keep original
                diff.append({"phrase": i, "action": "keep", "original": original, "result": original})
            else:
                diff.append({"phrase": i, "action": "rewrite", "original": original, "result": result})

        elif line.startswith("[PARAPHRASE"):
            result = _result_after_arrow(_after_bracket(line), fallback="")
            if not result:
                diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})
            elif len(result.split()) >= len(original.split()):
                diff.append({"phrase": i, "action": "keep", "original": original, "result": original})
            else:
                diff.append({"phrase": i, "action": "paraphrase", "original": original, "result": result})

        else:
            diff.append({"phrase": i, "action": "keep", "original": original, "result": original})

    return diff
