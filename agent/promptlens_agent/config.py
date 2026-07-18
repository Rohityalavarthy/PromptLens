"""Configuration file support for PromptLens CLI.

Supports .promptlensrc.toml and pyproject.toml [tool.promptlens] for persistent
configuration. CLI flags override config file values; env vars override file config.
"""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib  # Python 3.11+ stdlib
import os
import logging

logger = logging.getLogger("promptlens.config")


@dataclass
class PromptLensConfig:
    provider: str = "together"  # together | openai | anthropic
    model: str = ""  # empty = use provider default
    api_key_env: str = ""  # env var name for API key
    threshold: float = 0.15
    saliency_threshold: float = 0.15
    m_samples: int = 20
    semantic: bool = False
    test_inputs_file: str | None = None
    output_format: str = "terminal"  # terminal | json | sarif
    skip_dirs: list[str] = field(
        default_factory=lambda: [".venv", "venv", "node_modules", "__pycache__", ".git"]
    )
    verbose: bool = False


def load_config(start_dir: Path | None = None) -> PromptLensConfig:
    """Load config with precedence:

    .promptlensrc.toml (searched up from start_dir) > pyproject.toml [tool.promptlens] > defaults.

    Env vars override file config:
      PROMPTLENS_PROVIDER, PROMPTLENS_THRESHOLD, PROMPTLENS_FORMAT, PROMPTLENS_VERBOSE,
      PROMPTLENS_M_SAMPLES
    """
    if start_dir is None:
        start_dir = Path.cwd()

    config = PromptLensConfig()

    # Search for .promptlensrc.toml up from start_dir
    rc_path = _find_config_file(start_dir, ".promptlensrc.toml")
    if rc_path:
        logger.debug(f"Loading config from {rc_path}")
        config = _merge_toml(config, rc_path)
    else:
        # Try pyproject.toml [tool.promptlens]
        pyproject = _find_config_file(start_dir, "pyproject.toml")
        if pyproject:
            config = _merge_pyproject(config, pyproject)

    # Env var overrides
    config = _apply_env_overrides(config)
    return config


def _find_config_file(start_dir: Path, filename: str) -> Path | None:
    """Search up from start_dir for filename. Stop at filesystem root."""
    current = start_dir.resolve()
    while True:
        candidate = current / filename
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _merge_toml(config: PromptLensConfig, path: Path) -> PromptLensConfig:
    """Read TOML file and merge into config."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _apply_dict(config, data)


def _merge_pyproject(config: PromptLensConfig, path: Path) -> PromptLensConfig:
    """Read [tool.promptlens] section from pyproject.toml."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("tool", {}).get("promptlens", {})
    if not section:
        return config
    return _apply_dict(config, section)


def _apply_dict(config: PromptLensConfig, data: dict) -> PromptLensConfig:
    """Apply dict values to config fields."""
    field_map = {
        "provider": "provider",
        "model": "model",
        "api_key_env": "api_key_env",
        "threshold": "threshold",
        "saliency_threshold": "saliency_threshold",
        "m_samples": "m_samples",
        "semantic": "semantic",
        "test_inputs_file": "test_inputs_file",
        "output_format": "output_format",
        "format": "output_format",  # alias
        "skip_dirs": "skip_dirs",
        "verbose": "verbose",
    }
    for key, attr in field_map.items():
        if key in data:
            setattr(config, attr, data[key])
    return config


def _apply_env_overrides(config: PromptLensConfig) -> PromptLensConfig:
    """Apply PROMPTLENS_* env vars."""
    if val := os.environ.get("PROMPTLENS_PROVIDER"):
        config.provider = val
    if val := os.environ.get("PROMPTLENS_THRESHOLD"):
        config.threshold = float(val)
        config.saliency_threshold = float(val)
    if val := os.environ.get("PROMPTLENS_FORMAT"):
        config.output_format = val
    if val := os.environ.get("PROMPTLENS_VERBOSE"):
        config.verbose = val.lower() in ("1", "true", "yes")
    if val := os.environ.get("PROMPTLENS_M_SAMPLES"):
        config.m_samples = int(val)
    return config
