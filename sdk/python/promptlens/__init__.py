from .types import (
    SaliencyReport, SaliencyScore, CompressionResult,
    SimilarityMode, Phrase, RegionType
)
from .shapley import run_shapley
from .segmenter import segment_prompt
from .provider import (
    GenerationProvider, EmbeddingProvider, JudgeProvider, LLMProvider,
    TogetherProvider, OpenAIProvider, AnthropicProvider,
    configure_provider,
)

__all__ = [
    "run_shapley",
    "segment_prompt",
    "SaliencyReport",
    "SaliencyScore",
    "CompressionResult",
    "SimilarityMode",
    "Phrase",
    "RegionType",
    "GenerationProvider",
    "EmbeddingProvider",
    "JudgeProvider",
    "LLMProvider",
    "TogetherProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "configure_provider",
]

__version__ = "0.1.0"
