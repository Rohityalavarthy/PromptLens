from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RegionType(Enum):
    PLAIN = "plain"
    CODE_BLOCK = "code_block"
    JSON = "json"
    BULLETS = "bullets"
    XML_TAGGED = "xml_tagged"
    MARKDOWN = "markdown"


class SimilarityMode(Enum):
    STANDARD = "standard"   # trigram cosine
    SEMANTIC = "semantic"   # embedding + judge


@dataclass
class Region:
    type: RegionType
    text: str
    start_offset: int = 0


@dataclass
class Phrase:
    text: str
    index: int
    atomic: bool = False
    region_type: RegionType = RegionType.PLAIN
    tag_name: Optional[str] = None       # for XML phrases — used during reconstruction
    char_start: int = -1                 # start offset in original prompt (inclusive)
    char_end: int = -1                   # end offset in original prompt (exclusive)
    source_text: str = ""                # exact slice of original prompt[char_start:char_end]


@dataclass
class SaliencyScore:
    phrase: Phrase
    score: float                          # 0.0 to 1.0, normalised
    raw_shapley: float                    # unnormalised average marginal contribution
    disposition: str = "keep"            # keep | remove | merge | rewrite


@dataclass
class SaliencyReport:
    prompt: str
    phrases: list[Phrase]
    scores: list[SaliencyScore]
    token_count: int
    redundancy_fraction: float           # fraction of phrases scoring below threshold
    compression_candidate_tokens: int    # tokens in low-saliency phrases
    m_samples: int
    test_inputs_used: int
    confidence: float                    # average score stability across inputs


@dataclass
class CompressionResult:
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    token_delta: int
    validation_verdict: str              # "PASS" | "MARGINAL" | "REVIEW" | "FAIL"
    worst_case_divergence: float
    saliency_report: SaliencyReport
    diff: list[dict]                     # list of {phrase, action, original, compressed}
