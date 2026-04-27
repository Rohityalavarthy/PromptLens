import pytest
from promptlens.segmenter import segment_prompt, detect_regions
from promptlens.types import RegionType


def test_plain_text_splits_at_sentences():
    prompt = "You are a helpful assistant. Always respond in a friendly tone. Be concise and accurate in your answers."
    phrases = segment_prompt(prompt)
    assert len(phrases) >= 2
    assert all(p.region_type == RegionType.PLAIN for p in phrases)
    # Each phrase should contain part of the original text
    reconstructed = ' '.join(p.text for p in phrases)
    assert "helpful assistant" in reconstructed


def test_json_splits_per_top_level_key():
    prompt = '{"role": "assistant", "tone": "friendly", "format": "markdown"}'
    phrases = segment_prompt(prompt)
    assert len(phrases) == 3
    assert all(p.region_type == RegionType.JSON for p in phrases)
    texts = [p.text for p in phrases]
    assert any('"role"' in t for t in texts)
    assert any('"tone"' in t for t in texts)
    assert any('"format"' in t for t in texts)


def test_bullet_list_one_phrase_per_item():
    prompt = "- Always be polite\n- Never share personal data\n- Respond in the user's language"
    phrases = segment_prompt(prompt)
    assert len(phrases) == 3
    assert all(p.region_type == RegionType.BULLETS for p in phrases)
    assert "Always be polite" in phrases[0].text
    assert "Never share personal data" in phrases[1].text


def test_xml_one_phrase_per_tag_pair():
    prompt = "<instructions>Be helpful and concise.</instructions>\n<tone>Professional and warm.</tone>"
    phrases = segment_prompt(prompt)
    assert len(phrases) == 2
    assert all(p.region_type == RegionType.XML_TAGGED for p in phrases)
    assert phrases[0].tag_name == "instructions"
    assert phrases[1].tag_name == "tone"


def test_code_block_is_atomic():
    prompt = "Use this format:\n```python\ndef greet(name):\n    return f'Hello {name}'\n```"
    phrases = segment_prompt(prompt)
    code_phrases = [p for p in phrases if p.region_type == RegionType.CODE_BLOCK]
    assert len(code_phrases) == 1
    assert code_phrases[0].atomic is True


def test_phrase_indices_are_sequential():
    prompt = "You are an assistant. Be concise. Be accurate."
    phrases = segment_prompt(prompt)
    indices = [p.index for p in phrases]
    assert indices == list(range(len(phrases)))


def test_empty_prompt_returns_empty_list():
    phrases = segment_prompt("")
    assert phrases == []


def test_mixed_prompt_detects_multiple_region_types():
    prompt = (
        "You are a helpful assistant.\n"
        "- Always be polite\n"
        "- Be concise\n"
        '{"format": "json"}'
    )
    regions = detect_regions(prompt)
    region_types = {r.type for r in regions}
    assert RegionType.PLAIN in region_types
    assert RegionType.BULLETS in region_types
    assert RegionType.JSON in region_types
