import asyncio
import aiohttp
from promptlens import SaliencyReport, SaliencyScore
from promptlens.generator import generate

COMPRESSION_SYSTEM_PROMPT = """You are a prompt compression specialist.

You will receive a prompt split into phrases, each labelled KEEP or COMPRESS.

Rules:
- KEEP phrases: reproduce them EXACTLY, word for word. Do not touch them.
- COMPRESS phrases: apply ONE of these actions and label it:
  - REMOVE: if the phrase adds no unique instruction not covered by other phrases
  - MERGE: if the phrase partially overlaps with an adjacent phrase — combine them concisely
  - REWRITE: if the phrase is load-bearing but verbose — express the same instruction in fewer tokens

Output format — for each phrase on its own line:
[KEEP] <exact original text>
[REMOVE] <original> → (removed)
[MERGE with N] <original> → <merged text>
[REWRITE] <original> → <rewritten text>

Do not add explanations. Do not change KEEP phrases. Preserve meaning of all KEEP phrases exactly."""


async def compress_prompt(report: SaliencyReport) -> tuple[str, list[dict]]:
    """
    Produce a compressed prompt from a SaliencyReport.
    Returns (compressed_prompt_text, diff_list).

    diff_list entries: {phrase, action, original, result}
    """
    # Build the labelled phrase list for the LLM
    phrase_lines = []
    for score in report.scores:
        label = "KEEP" if score.disposition == "keep" else "COMPRESS"
        phrase_lines.append(f"[{label}] {score.phrase.text}")

    user_message = "Compress the following prompt phrases:\n\n" + "\n".join(phrase_lines)

    response = await generate(
        prompt=user_message,
        system_prompt=COMPRESSION_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=2048,
    )

    compressed_parts, diff = _parse_compression_response(response, report.scores)
    compressed_prompt = " ".join(compressed_parts)
    return compressed_prompt, diff


def _parse_compression_response(
    response: str,
    scores: list[SaliencyScore],
) -> tuple[list[str], list[dict]]:
    """
    Parse the labelled compression output back into a phrase list and diff.
    Falls back to original phrase text if parsing fails for a given line.
    """
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    compressed_parts = []
    diff = []

    for i, line in enumerate(lines):
        original = scores[i].phrase.text if i < len(scores) else ""

        if line.startswith("[KEEP]"):
            text = line[6:].strip()
            compressed_parts.append(text)
            diff.append({"phrase": i, "action": "keep", "original": original, "result": text})

        elif line.startswith("[REMOVE]"):
            diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})
            # Nothing added to compressed_parts

        elif line.startswith("[MERGE"):
            arrow = line.find("→")
            result = line[arrow+1:].strip() if arrow != -1 else original
            compressed_parts.append(result)
            diff.append({"phrase": i, "action": "merge", "original": original, "result": result})

        elif line.startswith("[REWRITE]"):
            arrow = line.find("→")
            result = line[arrow+1:].strip() if arrow != -1 else original
            compressed_parts.append(result)
            diff.append({"phrase": i, "action": "rewrite", "original": original, "result": result})

        else:
            # Unparseable line — keep original as fallback
            compressed_parts.append(original)
            diff.append({"phrase": i, "action": "keep", "original": original, "result": original})

    return compressed_parts, diff
