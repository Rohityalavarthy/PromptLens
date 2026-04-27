import pytest
from promptlens.similarity import trigram_similarity, get_trigrams


def test_identical_strings_score_zero():
    text = "You are a helpful assistant who responds concisely."
    assert trigram_similarity(text, text) == pytest.approx(0.0, abs=1e-6)


def test_completely_different_strings_score_high():
    a = "You are a helpful assistant."
    b = "XYZXYZXYZ ABCABCABC 123123123"
    score = trigram_similarity(a, b)
    assert score > 0.8


def test_similar_strings_score_low():
    a = "You are a helpful assistant. Always respond concisely."
    b = "You are a helpful assistant. Always respond briefly."
    score = trigram_similarity(a, b)
    # Should be similar (low divergence)
    assert score < 0.3


def test_empty_strings_score_zero():
    assert trigram_similarity("", "") == 0.0


def test_trigrams_extracted_correctly():
    tg = get_trigrams("abcde")
    assert "abc" in tg
    assert "bcd" in tg
    assert "cde" in tg
    assert len(tg) == 3
