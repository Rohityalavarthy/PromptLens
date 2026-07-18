import re
import json
from .types import Phrase, Region, RegionType


# ── Region Detection ──────────────────────────────────────────────────────────

def detect_regions(prompt: str) -> list[Region]:
    """
    Scan prompt top-to-bottom, classify contiguous text regions by structural type.
    Returns ordered list of regions. Every character belongs to exactly one region.
    Each region carries its start_offset within the original prompt.
    """
    regions = []
    lines = prompt.split('\n')
    i = 0
    # Track character position within the original prompt
    char_pos = 0
    # Precompute the starting character position of each line
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1  # +1 for the '\n'

    while i < len(lines):
        line = lines[i]

        # Code fence block — ```...```
        if line.strip().startswith('```'):
            block_lines = [line]
            block_start = line_starts[i]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                block_lines.append(lines[i])
                i += 1
            else:
                pass  # unclosed fence
            text = '\n'.join(block_lines)
            regions.append(Region(type=RegionType.CODE_BLOCK, text=text, start_offset=block_start))
            continue

        # JSON block — line starts with { or [
        if line.strip().startswith(('{', '[')):
            block_lines = [line]
            block_start = line_starts[i]
            i += 1
            depth = line.count('{') + line.count('[') - line.count('}') - line.count(']')
            while i < len(lines) and depth > 0:
                block_lines.append(lines[i])
                depth += lines[i].count('{') + lines[i].count('[')
                depth -= lines[i].count('}') + lines[i].count(']')
                i += 1
            text = '\n'.join(block_lines)
            regions.append(Region(type=RegionType.JSON, text=text, start_offset=block_start))
            continue

        # Bullet list — lines starting with -, *, •, or N.
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            block_lines = []
            block_start = line_starts[i]
            while i < len(lines) and (re.match(r'^\s*([-*•]|\d+\.)\s', lines[i]) or
                                       (block_lines and lines[i].startswith('  '))):
                block_lines.append(lines[i])
                i += 1
            text = '\n'.join(block_lines)
            regions.append(Region(type=RegionType.BULLETS, text=text, start_offset=block_start))
            continue

        # Markdown section — ## or ### header
        if re.match(r'^#{1,3}\s', line):
            block_lines = [line]
            block_start = line_starts[i]
            i += 1
            while i < len(lines) and not re.match(r'^#{1,3}\s', lines[i]):
                block_lines.append(lines[i])
                i += 1
            text = '\n'.join(block_lines)
            regions.append(Region(type=RegionType.MARKDOWN, text=text, start_offset=block_start))
            continue

        # XML tagged — line contains <tag>
        if re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line):
            block_lines = [line]
            block_start = line_starts[i]
            i += 1
            while i < len(lines) and re.search(r'</?[a-zA-Z_][a-zA-Z0-9_]*>', lines[i]):
                block_lines.append(lines[i])
                i += 1
            text = '\n'.join(block_lines)
            regions.append(Region(type=RegionType.XML_TAGGED, text=text, start_offset=block_start))
            continue

        # Plain text — collect until next structural boundary
        block_lines = [line]
        block_start = line_starts[i]
        i += 1
        while i < len(lines) and not _is_structural_start(lines[i]):
            block_lines.append(lines[i])
            i += 1
        text = '\n'.join(block_lines).strip()
        if text:
            # The stripped text may not start exactly at block_start if there was
            # leading whitespace/newlines. Find where the stripped text actually starts.
            raw_text = '\n'.join(block_lines)
            leading = len(raw_text) - len(raw_text.lstrip())
            regions.append(Region(type=RegionType.PLAIN, text=text, start_offset=block_start + leading))

    return [r for r in regions if r.text.strip()]


def _is_structural_start(line: str) -> bool:
    return bool(
        line.strip().startswith('```') or
        line.strip().startswith(('{', '[')) or
        re.match(r'^\s*([-*•]|\d+\.)\s', line) or
        re.match(r'^#{1,3}\s', line) or
        re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line)
    )


# ── Span computation helper ──────────────────────────────────────────────────

def _find_phrase_span(phrase_text: str, region_text: str, region_offset: int, search_start: int = 0) -> tuple[int, int, int]:
    """
    Find the character span of phrase_text within the original prompt.
    Returns (char_start, char_end, next_search_start).
    If not found, returns (-1, -1, search_start).
    """
    pos = region_text.find(phrase_text, search_start)
    if pos != -1:
        char_start = region_offset + pos
        char_end = char_start + len(phrase_text)
        return char_start, char_end, pos + len(phrase_text)
    return -1, -1, search_start


# ── Per-type segmenters ────────────────────────────────────────────────────────

def segment_plain_text(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """
    Split on sentence boundaries (. ! ?) then clause boundaries (, ; :).
    Apply 35-60 char heuristic. Non-atomic.
    """
    MIN_CHARS, MAX_CHARS = 35, 60
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    phrases = []
    idx = start_index
    search_start = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHARS:
            phrase = Phrase(text=sentence, index=idx, atomic=False, region_type=RegionType.PLAIN)
            if region_offset >= 0:
                cs, ce, search_start = _find_phrase_span(sentence, text, region_offset, search_start)
                phrase.char_start = cs
                phrase.char_end = ce
                if cs >= 0:
                    phrase.source_text = sentence  # plain text: no transformation
            phrases.append(phrase)
            idx += 1
        else:
            clauses = re.split(r'(?<=[,;:])\s+', sentence)
            current = ''
            for clause in clauses:
                if len(current) + len(clause) < MAX_CHARS:
                    current = (current + ' ' + clause).strip()
                else:
                    if len(current) >= MIN_CHARS:
                        phrase = Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN)
                        if region_offset >= 0:
                            cs, ce, search_start = _find_phrase_span(current, text, region_offset, search_start)
                            phrase.char_start = cs
                            phrase.char_end = ce
                            if cs >= 0:
                                phrase.source_text = current
                        phrases.append(phrase)
                        idx += 1
                        current = clause
                    else:
                        current = (current + ' ' + clause).strip()
            if current:
                phrase = Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN)
                if region_offset >= 0:
                    cs, ce, search_start = _find_phrase_span(current, text, region_offset, search_start)
                    phrase.char_start = cs
                    phrase.char_end = ce
                    if cs >= 0:
                        phrase.source_text = current
                phrases.append(phrase)
                idx += 1

    return phrases


def segment_code_block(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """Entire code block is one atomic phrase."""
    phrase = Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.CODE_BLOCK)
    if region_offset >= 0:
        phrase.char_start = region_offset
        phrase.char_end = region_offset + len(text)
        phrase.source_text = text
    return [phrase]


def segment_json(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """
    Parse JSON and treat each top-level key-value pair as one phrase.
    Fall back to atomic on parse failure.
    """
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            phrase = Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)
            if region_offset >= 0:
                phrase.char_start = region_offset
                phrase.char_end = region_offset + len(text)
                phrase.source_text = text
            return [phrase]
        phrases = []
        search_start = 0
        for i, (key, value) in enumerate(parsed.items()):
            phrase_text = f'"{key}": {json.dumps(value)}'
            phrase = Phrase(text=phrase_text, index=start_index + i, atomic=False, region_type=RegionType.JSON)
            if region_offset >= 0:
                # Find the key in the original text to get source span
                key_pattern = f'"{key}"'
                key_pos = text.find(key_pattern, search_start)
                if key_pos != -1:
                    # Find the end of this value — look for next key or end of object
                    # Simple heuristic: find next key or closing brace
                    remaining_keys = list(parsed.keys())[i + 1:]
                    if remaining_keys:
                        next_key = f'"{remaining_keys[0]}"'
                        next_pos = text.find(next_key, key_pos + len(key_pattern))
                        # go back to find the comma separator
                        if next_pos != -1:
                            # source ends before the comma+whitespace before next key
                            end_pos = next_pos
                            # trim trailing comma and whitespace
                            while end_pos > key_pos and text[end_pos - 1] in ' ,\n\t\r':
                                end_pos -= 1
                            phrase.char_start = region_offset + key_pos
                            phrase.char_end = region_offset + end_pos
                            phrase.source_text = text[key_pos:end_pos]
                            search_start = next_pos
                        else:
                            phrase.char_start = region_offset + key_pos
                            phrase.char_end = region_offset + len(text)
                            phrase.source_text = text[key_pos:]
                    else:
                        # Last key — ends before closing brace
                        end_pos = text.rfind('}')
                        if end_pos == -1:
                            end_pos = len(text)
                        # trim trailing whitespace/newlines before }
                        while end_pos > key_pos and text[end_pos - 1] in ' \n\t\r':
                            end_pos -= 1
                        phrase.char_start = region_offset + key_pos
                        phrase.char_end = region_offset + end_pos
                        phrase.source_text = text[key_pos:end_pos]
            phrases.append(phrase)
        return phrases
    except (json.JSONDecodeError, ValueError):
        phrase = Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)
        if region_offset >= 0:
            phrase.char_start = region_offset
            phrase.char_end = region_offset + len(text)
            phrase.source_text = text
        return [phrase]


def segment_bullets(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """Each bullet item (including its continuation lines) is one phrase."""
    phrases = []
    idx = start_index
    current_bullet = None
    current_bullet_start = 0
    search_start = 0

    for line in text.split('\n'):
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            if current_bullet:
                phrase = Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS)
                if region_offset >= 0:
                    # Find the original bullet text in the region
                    cs, ce, search_start = _find_phrase_span(current_bullet.strip(), text, region_offset, current_bullet_start)
                    phrase.char_start = cs
                    phrase.char_end = ce
                    if cs >= 0:
                        phrase.source_text = current_bullet.strip()
                phrases.append(phrase)
                idx += 1
            current_bullet = line
            current_bullet_start = search_start
        elif current_bullet and line.startswith('  '):
            current_bullet += ' ' + line.strip()

    if current_bullet:
        phrase = Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS)
        if region_offset >= 0:
            cs, ce, search_start = _find_phrase_span(current_bullet.strip(), text, region_offset, current_bullet_start)
            phrase.char_start = cs
            phrase.char_end = ce
            if cs >= 0:
                phrase.source_text = current_bullet.strip()
        phrases.append(phrase)

    return phrases


def segment_xml(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """Each <tag>content</tag> pair is one phrase. Preserves tag_name for reconstruction."""
    pattern = re.compile(r'<(\w+)>([\s\S]*?)</\1>', re.DOTALL)
    phrases = []
    idx = start_index

    for match in pattern.finditer(text):
        phrase = Phrase(
            text=match.group(0),
            index=idx,
            atomic=False,
            region_type=RegionType.XML_TAGGED,
            tag_name=match.group(1),
        )
        if region_offset >= 0:
            phrase.char_start = region_offset + match.start()
            phrase.char_end = region_offset + match.end()
            phrase.source_text = match.group(0)
        phrases.append(phrase)
        idx += 1

    if not phrases:
        return segment_plain_text(text, start_index, region_offset=region_offset)
    return phrases


def segment_markdown(text: str, start_index: int = 0, region_offset: int = -1) -> list[Phrase]:
    """
    Split at ## / ### headers. Header is atomic. Body uses plain text segmenter.
    """
    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    phrases = []
    idx = start_index
    search_start = 0

    for section in sections:
        if not section.strip():
            continue
        header_match = re.match(r'^(#{1,3}\s.+)\n?', section)
        if header_match:
            header_text = header_match.group(1).strip()
            phrase = Phrase(
                text=header_text,
                index=idx,
                atomic=True,
                region_type=RegionType.MARKDOWN,
            )
            if region_offset >= 0:
                cs, ce, search_start = _find_phrase_span(header_text, text, region_offset, search_start)
                phrase.char_start = cs
                phrase.char_end = ce
                if cs >= 0:
                    phrase.source_text = header_text
            phrases.append(phrase)
            idx += 1
            body = section[len(header_match.group(0)):]
            if body.strip():
                # Find body offset within the region text
                body_offset = -1
                if region_offset >= 0:
                    body_stripped = body.strip()
                    body_pos = text.find(body_stripped, search_start)
                    if body_pos != -1:
                        body_offset = region_offset + body_pos
                        search_start = body_pos + len(body_stripped)
                body_phrases = segment_plain_text(body.strip(), idx, region_offset=body_offset)
                phrases.extend(body_phrases)
                idx += len(body_phrases)
        else:
            section_offset = -1
            if region_offset >= 0:
                sec_stripped = section.strip()
                sec_pos = text.find(sec_stripped, search_start)
                if sec_pos != -1:
                    section_offset = region_offset + sec_pos
                    search_start = sec_pos + len(sec_stripped)
            plain = segment_plain_text(section.strip(), idx, region_offset=section_offset)
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
        region_phrases = fn(region.text, start_index=idx, region_offset=region.start_offset)
        phrases.extend(region_phrases)
        idx += len(region_phrases)

    return phrases
