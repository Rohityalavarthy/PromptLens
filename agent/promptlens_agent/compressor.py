import re
import warnings
import logging
from promptlens import SaliencyReport, SaliencyScore
from promptlens.generator import generate

logger = logging.getLogger(__name__)


def reconstruct_from_original(
    original_prompt: str,
    diff: list[dict],
    scores: list[SaliencyScore] | None = None,
) -> str:
    """
    Reconstruct a compressed prompt by substituting phrases directly in the
    original text. Preserves all whitespace, newlines, and structure.

    If scores are provided and have valid char_start/char_end, uses offset-based
    reconstruction (processes end-to-start to preserve earlier offsets).
    Otherwise falls back to legacy str.replace(old, new, 1) approach.
    """
    # Determine whether to use offset-based reconstruction
    use_offsets = (
        scores is not None
        and len(scores) > 0
        and scores[0].phrase.char_start != -1
    )

    if use_offsets:
        result = _reconstruct_offset_based(original_prompt, diff, scores)
    else:
        if scores is not None and len(scores) > 0 and scores[0].phrase.char_start == -1:
            warnings.warn(
                "Phrase spans not populated; falling back to legacy str.replace reconstruction. "
                "Re-run segmentation to enable offset-based reconstruction.",
                stacklevel=2,
            )
        result = _reconstruct_legacy(original_prompt, diff)

    # Clean up artifacts from removals
    result = re.sub(r'\n{3,}', '\n\n', result)       # multiple blank lines → one
    result = re.sub(r'\n[ \t]+\n', '\n\n', result)   # whitespace-only lines
    result = re.sub(r'[ \t]+\n', '\n', result)        # trailing spaces on lines

    return result.strip()


def _reconstruct_offset_based(
    original_prompt: str,
    diff: list[dict],
    scores: list[SaliencyScore],
) -> str:
    """Offset-based reconstruction: process diffs end-to-start to preserve offsets."""
    # Build list of (char_start, char_end, replacement) sorted by char_start descending
    edits = []
    for entry in diff:
        if entry["action"] == "keep":
            continue
        phrase_idx = entry.get("phrase", -1)
        if 0 <= phrase_idx < len(scores):
            phrase = scores[phrase_idx].phrase
            if phrase.char_start >= 0 and phrase.char_end >= 0:
                edits.append((phrase.char_start, phrase.char_end, entry["result"]))
                continue
        # Fallback for this particular entry if no valid offset
        # (shouldn't happen in offset mode, but be safe)
        pass

    # Sort by start position descending so earlier offsets remain valid
    edits.sort(key=lambda x: x[0], reverse=True)

    result = original_prompt
    for start, end, replacement in edits:
        result = result[:start] + replacement + result[end:]

    return result


def _reconstruct_legacy(original_prompt: str, diff: list[dict]) -> str:
    """Legacy str.replace-based reconstruction."""
    result = original_prompt

    for entry in diff:
        if entry["action"] == "keep":
            continue
        old = entry["original"]
        new = entry["result"]  # "" for remove, replacement text for all others
        if old and old in result:
            result = result.replace(old, new, 1)

    return result


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

## Output format — one line per input phrase, using its phrase ID
  [#0][KEEP] <exact original text>
  [#1][REMOVE] <original text>
  [#2][REWRITE] <original text> → <rewritten result>
  [#3][MERGE into #1] <original text>
  [#4][PARAPHRASE] <original text> → <paraphrased result>

## Examples
Input:
  [#0][KEEP:0.91] Always respond concisely.
  [#1][COMPRESS:0.03] Be brief in your answers.
  [#2][COMPRESS:0.08] Keep responses short and to the point.

Output:
  [#0][KEEP] Always respond concisely.
  [#1][REMOVE] Be brief in your answers.
  [#2][MERGE into #0] Keep responses short and to the point.

Input:
  [#0][KEEP:0.85] You are a helpful coding assistant.
  [#1][COMPRESS:0.04] You help users write better code and fix bugs.
  [#2][COMPRESS:0.12] Always include code examples in your responses when relevant.

Output:
  [#0][KEEP] You are a helpful coding assistant.
  [#1][REMOVE] You help users write better code and fix bugs.
  [#2][REWRITE] Always include code examples in your responses when relevant. → Include code examples when relevant.

## Hard rules
  1. KEEP phrases must be reproduced EXACTLY — not one word changed.
  2. REWRITE, MERGE, and PARAPHRASE results must never be empty — use [REMOVE] if truly \
nothing unique remains.
  3. Every input phrase must appear exactly once in the output, using its [#N] ID.
  4. No explanations or commentary.
  5. MERGE target must be a valid phrase ID (0-based)."""


async def compress_prompt(report: SaliencyReport, threshold: float = 0.15) -> tuple[str, list[dict]]:
    """
    Produce a compressed prompt from a SaliencyReport.
    Returns (compressed_prompt_text, diff_list).

    diff_list entries: {phrase, action, original, result}
    On malformed response (coverage < 0.8), retries once with feedback.
    """
    redundancy_pct = (
        int(report.compression_candidate_tokens / report.token_count * 100)
        if report.token_count > 0 else 0
    )

    phrase_lines = []
    for score in report.scores:
        label = "KEEP" if score.disposition == "keep" else "COMPRESS"
        phrase_lines.append(f"[#{score.phrase.index}][{label}:{score.score:.2f}] {score.phrase.text}")

    user_message = "Compress the following prompt phrases:\n\n" + "\n".join(phrase_lines)
    system_prompt = _build_compression_prompt(threshold, redundancy_pct)

    max_attempts = 2
    for attempt in range(max_attempts):
        response = await generate(
            prompt=user_message,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=2048,
        )

        diff = _parse_compression_response(response, report.scores)
        coverage = _validate_coverage(diff, report.scores)

        if coverage >= 0.8 or attempt == max_attempts - 1:
            break

        # Retry with feedback
        user_message = (
            f"Your previous response only covered {coverage:.0%} of the phrases. "
            f"Please output one line per phrase using its [#N] ID prefix. "
            f"Every phrase must appear exactly once.\n\n"
            "Compress the following prompt phrases:\n\n" + "\n".join(phrase_lines)
        )

    # Apply MERGE semantics
    diff = _apply_merge_semantics(diff, report.scores)

    compressed_prompt = reconstruct_from_original(report.prompt, diff, report.scores)
    return compressed_prompt, diff


def _validate_coverage(diff: list[dict], scores: list[SaliencyScore]) -> float:
    """
    Returns fraction of scores that have a corresponding diff entry.
    """
    if not scores:
        return 1.0
    matched_ids = {entry["phrase"] for entry in diff}
    covered = sum(1 for s in scores if s.phrase.index in matched_ids)
    return covered / len(scores)


def _apply_merge_semantics(diff: list[dict], scores: list[SaliencyScore]) -> list[dict]:
    """
    Process MERGE entries:
    - [#N][MERGE into #M] means phrase N is removed, content incorporated into M
    - If M is KEEP, change M to REWRITE with merged result
    - Multiple merges into same target: allowed
    - Merge target already removed: log warning, treat as standalone REMOVE
    - Merge target already merged into another: log warning, treat as REMOVE
    """
    # Build index of diff entries by phrase ID
    diff_by_id = {entry["phrase"]: entry for entry in diff}
    num_phrases = len(scores)

    # Collect merge entries
    merge_entries = [(i, entry) for i, entry in enumerate(diff) if entry["action"] == "merge"]

    for _, entry in merge_entries:
        merge_target = entry.get("merge_target")
        if merge_target is None:
            continue

        # Validate merge target
        if merge_target < 0 or merge_target >= num_phrases:
            _demote_to_remove(
                entry, f"Merge target #{merge_target} out of range for phrase #{entry['phrase']}; treating as REMOVE"
            )
            continue

        if merge_target == entry["phrase"]:
            _demote_to_remove(
                entry, f"Self-referencing merge for phrase #{entry['phrase']}; treating as REMOVE"
            )
            continue

        target_entry = diff_by_id.get(merge_target)
        if target_entry is None:
            # Target not in diff — add a default KEEP entry? Just treat as remove.
            _demote_to_remove(
                entry, f"Merge target #{merge_target} not found in diff for phrase #{entry['phrase']}; treating as REMOVE"
            )
            continue

        # Check if target is already removed
        if target_entry["action"] == "remove":
            _demote_to_remove(
                entry, f"Merge target #{merge_target} is already removed for phrase #{entry['phrase']}; treating as REMOVE"
            )
            continue

        # Check if target is itself merged into another
        if target_entry["action"] == "merge" and "merge_target" in target_entry:
            _demote_to_remove(
                entry, f"Merge target #{merge_target} is itself merged into another for phrase #{entry['phrase']}; treating as REMOVE"
            )
            continue

        # Valid merge: source is removed, target absorbs content
        # Entry for the source phrase
        entry["result"] = ""

        # If target is currently KEEP, change it to REWRITE with merged text
        if target_entry["action"] == "keep":
            merged_text = target_entry["original"] + " " + entry["original"]
            target_entry["action"] = "rewrite"
            target_entry["result"] = merged_text.strip()

    return diff


def _demote_to_remove(entry: dict, reason: str) -> None:
    """Convert a merge entry to a remove when the merge target is invalid."""
    logger.warning(reason)
    entry["action"] = "remove"
    entry["result"] = ""
    entry.pop("merge_target", None)


def _after_bracket(line: str) -> str:
    """Return the text that follows the first closing ] on a line."""
    close = line.find("]")
    return line[close + 1:].strip() if close != -1 else ""


def _after_action_bracket(line: str) -> str:
    """Return text after the action bracket [ACTION], accounting for [#N][ACTION] prefix."""
    # Find the action bracket (second ] if line starts with [#)
    if line.startswith("[#"):
        first_close = line.find("]")
        if first_close != -1:
            second_close = line.find("]", first_close + 1)
            if second_close != -1:
                return line[second_close + 1:].strip()
    # Fallback: after first ]
    close = line.find("]")
    return line[close + 1:].strip() if close != -1 else ""


def _result_after_arrow(text: str, fallback: str) -> str:
    """Extract the result half of 'original → result'. Returns fallback if no arrow."""
    arrow = text.find("→")
    return text[arrow + 1:].strip() if arrow != -1 else fallback


# Regex for ID-based format: [#N][ACTION] or [#N][MERGE into #M]
_ID_PATTERN = re.compile(r'\[#(\d+)\]\[(\w+)(?:\s+into\s+#(\d+))?\]')


def _parse_compression_response(
    response: str,
    scores: list[SaliencyScore],
) -> list[dict]:
    """
    Parse the labelled compression output into a diff list.

    Strategy (ordered by priority):
    1. Primary: Parse by phrase ID — regex extracts (phrase_id, action, merge_target)
    2. Fallback: Content matching — if no valid [#N] prefix, attempt text comparison
    3. Default: KEEP — any score not matched defaults to keep

    Handles both old format ([KEEP] text) and new ID-based format ([#N][KEEP] text).
    """
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    diff = []
    matched_ids: set[int] = set()

    # First pass: try ID-based parsing
    for line in lines:
        id_match = _ID_PATTERN.match(line)
        if id_match:
            phrase_id = int(id_match.group(1))
            action_str = id_match.group(2).upper()
            merge_target = int(id_match.group(3)) if id_match.group(3) else None

            # Skip duplicate IDs — take first occurrence
            if phrase_id in matched_ids:
                continue

            if phrase_id < 0 or phrase_id >= len(scores):
                continue  # out of range, skip

            matched_ids.add(phrase_id)
            original = scores[phrase_id].phrase.text
            text_after = _after_action_bracket(line)
            entry = _parse_action(action_str, text_after, original, phrase_id, merge_target)
            diff.append(entry)
        else:
            # Try old-style format (positional, no ID)
            entry = _parse_old_format_line(line, len(diff), scores)
            if entry is not None:
                phrase_id = entry["phrase"]
                if phrase_id not in matched_ids:
                    matched_ids.add(phrase_id)
                    diff.append(entry)

    # If we parsed via old-format (positional) and got results, we're done
    # Otherwise, attempt content-match fallback for any lines that didn't parse
    if not diff:
        # Pure fallback to old positional parsing
        return _parse_old_format(response, scores)

    # Fill in unmatched scores with content-match fallback, then default to KEEP
    unmatched_lines = []
    for line in lines:
        id_match = _ID_PATTERN.match(line)
        if not id_match and not line.startswith("["):
            unmatched_lines.append(line)

    for score in scores:
        if score.phrase.index not in matched_ids:
            # Try content matching against unmatched lines
            found = False
            for uline in unmatched_lines:
                normalized_line = uline.strip().lower()
                normalized_phrase = score.phrase.text.strip().lower()
                if normalized_phrase in normalized_line or normalized_line in normalized_phrase:
                    matched_ids.add(score.phrase.index)
                    diff.append({
                        "phrase": score.phrase.index,
                        "action": "keep",
                        "original": score.phrase.text,
                        "result": score.phrase.text,
                    })
                    unmatched_lines.remove(uline)
                    found = True
                    break
            if not found:
                # Default: KEEP
                diff.append({
                    "phrase": score.phrase.index,
                    "action": "keep",
                    "original": score.phrase.text,
                    "result": score.phrase.text,
                })

    # Sort diff by phrase index for consistency
    diff.sort(key=lambda x: x["phrase"])
    return diff


def _parse_action(action_str: str, text_after: str, original: str, phrase_id: int, merge_target: int | None) -> dict:
    """Parse a single action into a diff entry."""
    if action_str == "KEEP":
        return {"phrase": phrase_id, "action": "keep", "original": original, "result": text_after or original}

    elif action_str == "REMOVE":
        return {"phrase": phrase_id, "action": "remove", "original": original, "result": ""}

    elif action_str == "MERGE":
        result = _result_after_arrow(text_after, fallback="")
        entry = {"phrase": phrase_id, "action": "merge", "original": original, "result": result, "merge_target": merge_target}
        if not result and merge_target is None:
            entry["action"] = "remove"
            entry["result"] = ""
        return entry

    elif action_str == "REWRITE":
        result = _result_after_arrow(text_after, fallback="")
        if not result:
            return {"phrase": phrase_id, "action": "remove", "original": original, "result": ""}
        elif len(result.split()) >= len(original.split()):
            return {"phrase": phrase_id, "action": "keep", "original": original, "result": original}
        else:
            return {"phrase": phrase_id, "action": "rewrite", "original": original, "result": result}

    elif action_str == "PARAPHRASE":
        result = _result_after_arrow(text_after, fallback="")
        if not result:
            return {"phrase": phrase_id, "action": "remove", "original": original, "result": ""}
        elif len(result.split()) >= len(original.split()):
            return {"phrase": phrase_id, "action": "keep", "original": original, "result": original}
        else:
            return {"phrase": phrase_id, "action": "paraphrase", "original": original, "result": result}

    else:
        # Unknown action — default to keep
        return {"phrase": phrase_id, "action": "keep", "original": original, "result": original}


def _parse_old_format_line(line: str, position: int, scores: list[SaliencyScore]) -> dict | None:
    """Parse a single line in the old format (no [#N] prefix). Returns None if not parseable."""
    original = scores[position].phrase.text if position < len(scores) else ""

    if line.startswith("[KEEP"):
        text = _after_bracket(line)
        return {"phrase": position, "action": "keep", "original": original, "result": text}

    elif line.startswith("[REMOVE"):
        return {"phrase": position, "action": "remove", "original": original, "result": ""}

    elif line.startswith("[MERGE"):
        result = _result_after_arrow(_after_bracket(line), fallback="")
        if not result:
            return {"phrase": position, "action": "remove", "original": original, "result": ""}
        else:
            return {"phrase": position, "action": "merge", "original": original, "result": result}

    elif line.startswith("[REWRITE"):
        result = _result_after_arrow(_after_bracket(line), fallback="")
        if not result:
            return {"phrase": position, "action": "remove", "original": original, "result": ""}
        elif len(result.split()) >= len(original.split()):
            return {"phrase": position, "action": "keep", "original": original, "result": original}
        else:
            return {"phrase": position, "action": "rewrite", "original": original, "result": result}

    elif line.startswith("[PARAPHRASE"):
        result = _result_after_arrow(_after_bracket(line), fallback="")
        if not result:
            return {"phrase": position, "action": "remove", "original": original, "result": ""}
        elif len(result.split()) >= len(original.split()):
            return {"phrase": position, "action": "keep", "original": original, "result": original}
        else:
            return {"phrase": position, "action": "paraphrase", "original": original, "result": result}

    else:
        return None


def _parse_old_format(response: str, scores: list[SaliencyScore]) -> list[dict]:
    """
    Legacy parser: parse the response using positional matching.
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
