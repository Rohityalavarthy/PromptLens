"""
Shapley tests that don't require a live API key.
Tests the pure computation logic: build_prompt, CoalitionCache, normalisation.
"""
import pytest
from promptlens.shapley import build_prompt, build_empty_prompt, CoalitionCache
from promptlens.types import Phrase, RegionType


def make_phrase(text: str, index: int) -> Phrase:
    return Phrase(text=text, index=index, atomic=False, region_type=RegionType.PLAIN)


def test_build_prompt_active_subset():
    phrases = [make_phrase("alpha", 0), make_phrase("beta", 1), make_phrase("gamma", 2)]
    result = build_prompt(phrases, {0, 2})
    assert result == "alpha gamma"


def test_build_prompt_empty_coalition():
    phrases = [make_phrase("alpha", 0), make_phrase("beta", 1)]
    result = build_prompt(phrases, set())
    assert result == ""


def test_build_empty_prompt_returns_placeholders():
    phrases = [make_phrase("alpha", 0), make_phrase("beta", 1), make_phrase("gamma", 2)]
    result = build_empty_prompt(phrases)
    assert result == "[...] [...] [...]"


def test_coalition_cache_stores_and_retrieves():
    cache = CoalitionCache()
    cache.set({0, 1, 2}, "some output")
    assert cache.get({0, 1, 2}) == "some output"
    assert cache.get({2, 0, 1}) == "some output"  # order-independent


def test_coalition_cache_miss_returns_none():
    cache = CoalitionCache()
    assert cache.get({0, 1}) is None


def test_coalition_cache_key_is_sorted():
    cache = CoalitionCache()
    assert cache.key({3, 1, 2}) == "1,2,3"
