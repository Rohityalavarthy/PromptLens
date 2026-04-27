import re
import json
from .types import Phrase, Region, RegionType


# ── Region Detection ──────────────────────────────────────────────────────────

def detect_regions(prompt: str) -> list[Region]:
    """
    Scan prompt top-to-bottom, classify contiguous text regions by structural type.
    Returns ordered list of regions. Every character belongs to exactly one region.
    """
    regions = []
    lines = prompt.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code fence block — ```...```
        if line.strip().startswith('```'):
            block_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                block_lines.append(lines[i])
            regions.append(Region(type=RegionType.CODE_BLOCK, text='\n'.join(block_lines)))
            i += 1
            continue

        # JSON block — line starts with { or [
        if line.strip().startswith(('{', '[')):
            block_lines = [line]
            i += 1
            depth = line.count('{') + line.count('[') - line.count('}') - line.count(']')
            while i < len(lines) and depth > 0:
                block_lines.append(lines[i])
                depth += lines[i].count('{') + lines[i].count('[')
                depth -= lines[i].count('}') + lines[i].count(']')
                i += 1
            regions.append(Region(type=RegionType.JSON, text='\n'.join(block_lines)))
            continue

        # Bullet list — lines starting with -, *, •, or N.
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            block_lines = []
            while i < len(lines) and (re.match(r'^\s*([-*•]|\d+\.)\s', lines[i]) or
                                       (block_lines and lines[i].startswith('  '))):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.BULLETS, text='\n'.join(block_lines)))
            continue

        # Markdown section — ## or ### header
        if re.match(r'^#{1,3}\s', line):
            block_lines = [line]
            i += 1
            while i < len(lines) and not re.match(r'^#{1,3}\s', lines[i]):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.MARKDOWN, text='\n'.join(block_lines)))
            continue

        # XML tagged — line contains <tag>
        if re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line):
            block_lines = [line]
            i += 1
            while i < len(lines) and re.search(r'</?[a-zA-Z_][a-zA-Z0-9_]*>', lines[i]):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.XML_TAGGED, text='\n'.join(block_lines)))
            continue

        # Plain text — collect until next structural boundary
        block_lines = [line]
        i += 1
        while i < len(lines) and not _is_structural_start(lines[i]):
            block_lines.append(lines[i])
            i += 1
        text = '\n'.join(block_lines).strip()
        if text:
            regions.append(Region(type=RegionType.PLAIN, text=text))

    return [r for r in regions if r.text.strip()]


def _is_structural_start(line: str) -> bool:
    return bool(
        line.strip().startswith('```') or
        line.strip().startswith(('{', '[')) or
        re.match(r'^\s*([-*•]|\d+\.)\s', line) or
        re.match(r'^#{1,3}\s', line) or
        re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line)
    )


# ── Per-type segmenters ────────────────────────────────────────────────────────

def segment_plain_text(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Split on sentence boundaries (. ! ?) then clause boundaries (, ; :).
    Apply 35-60 char heuristic. Non-atomic.
    """
    MIN_CHARS, MAX_CHARS = 35, 60
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    phrases = []
    idx = start_index

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHARS:
            phrases.append(Phrase(text=sentence, index=idx, atomic=False, region_type=RegionType.PLAIN))
            idx += 1
        else:
            clauses = re.split(r'(?<=[,;:])\s+', sentence)
            current = ''
            for clause in clauses:
                if len(current) + len(clause) < MAX_CHARS:
                    current = (current + ' ' + clause).strip()
                else:
                    if len(current) >= MIN_CHARS:
                        phrases.append(Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN))
                        idx += 1
                        current = clause
                    else:
                        current = (current + ' ' + clause).strip()
            if current:
                phrases.append(Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN))
                idx += 1

    return phrases


def segment_code_block(text: str, start_index: int = 0) -> list[Phrase]:
    """Entire code block is one atomic phrase."""
    return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.CODE_BLOCK)]


def segment_json(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Parse JSON and treat each top-level key-value pair as one phrase.
    Fall back to atomic on parse failure.
    """
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)]
        phrases = []
        for i, (key, value) in enumerate(parsed.items()):
            phrase_text = f'"{key}": {json.dumps(value)}'
            phrases.append(Phrase(text=phrase_text, index=start_index + i, atomic=False, region_type=RegionType.JSON))
        return phrases
    except (json.JSONDecodeError, ValueError):
        return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)]


def segment_bullets(text: str, start_index: int = 0) -> list[Phrase]:
    """Each bullet item (including its continuation lines) is one phrase."""
    phrases = []
    idx = start_index
    current_bullet = None

    for line in text.split('\n'):
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            if current_bullet:
                phrases.append(Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS))
                idx += 1
            current_bullet = line
        elif current_bullet and line.startswith('  '):
            current_bullet += ' ' + line.strip()

    if current_bullet:
        phrases.append(Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS))

    return phrases


def segment_xml(text: str, start_index: int = 0) -> list[Phrase]:
    """Each <tag>content</tag> pair is one phrase. Preserves tag_name for reconstruction."""
    pattern = re.compile(r'<(\w+)>([\s\S]*?)</\1>', re.DOTALL)
    phrases = []
    idx = start_index

    for match in pattern.finditer(text):
        phrases.append(Phrase(
            text=match.group(0),
            index=idx,
            atomic=False,
            region_type=RegionType.XML_TAGGED,
            tag_name=match.group(1),
        ))
        idx += 1

    if not phrases:
        return segment_plain_text(text, start_index)
    return phrases


def segment_markdown(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Split at ## / ### headers. Header is atomic. Body uses plain text segmenter.
    """
    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    phrases = []
    idx = start_index

    for section in sections:
        if not section.strip():
            continue
        header_match = re.match(r'^(#{1,3}\s.+)\n?', section)
        if header_match:
            phrases.append(Phrase(
                text=header_match.group(1).strip(),
                index=idx,
                atomic=True,
                region_type=RegionType.MARKDOWN,
            ))
            idx += 1
            body = section[len(header_match.group(0)):]
            if body.strip():
                body_phrases = segment_plain_text(body.strip(), idx)
                phrases.extend(body_phrases)
                idx += len(body_phrases)
        else:
            plain = segment_plain_text(section.strip(), idx)
            phrases.extend(plain)
            idx += len(plain)

    return phrases


# ── Main entry point ──────────────────────────────────────────────────────────

def segment_prompt(prompt: str) -> list[Phrase]:
    """
    Full pipeline: detect regions → route to segmenter → return flat phrase list.
    Atomic phrases are never split by character count.
    """
    regions = detect_regions(prompt)
    phrases = []
    idx = 0

    segmenters = {
        RegionType.PLAIN:      segment_plain_text,
        RegionType.CODE_BLOCK: segment_code_block,
        RegionType.JSON:       segment_json,
        RegionType.BULLETS:    segment_bullets,
        RegionType.XML_TAGGED: segment_xml,
        RegionType.MARKDOWN:   segment_markdown,
    }

    for region in regions:
        fn = segmenters.get(region.type, segment_plain_text)
        region_phrases = fn(region.text, start_index=idx)
        phrases.extend(region_phrases)
        idx += len(region_phrases)

    return phrases
