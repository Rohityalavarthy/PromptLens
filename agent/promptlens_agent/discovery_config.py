import yaml
from pathlib import Path
from .discovery import DiscoveredPrompt

PROMPT_KEYS = {"system_prompt", "prompt", "template", "system_message", "system", "instructions", "system_template"}
TEMPLATE_DIRS = {"prompts", "templates", "prompt_templates"}
TEMPLATE_EXTENSIONS = {".jinja2", ".j2", ".prompt"}
CONFIG_EXTENSIONS = {".yaml", ".yml"}
SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", "dist", "build"}


def discover_prompts_config(repo_path: str) -> list[DiscoveredPrompt]:
    """Scan YAML config files and template directories for prompts."""
    discovered = []
    repo = Path(repo_path)

    # 1. Scan YAML files for prompt-like keys
    for config_file in repo.rglob("*"):
        if config_file.suffix not in CONFIG_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in config_file.parts):
            continue
        if not config_file.is_file():
            continue
        discovered.extend(_scan_yaml_file(config_file))

    # 2. Scan template directories
    for entry in repo.rglob("*"):
        if not entry.is_dir():
            continue
        if entry.name.lower() in TEMPLATE_DIRS or "prompt" in entry.name.lower():
            discovered.extend(_scan_template_dir(entry))

    return discovered


def _scan_yaml_file(path: Path) -> list[DiscoveredPrompt]:
    """Recursively walk YAML dict looking for prompt-like keys with string values > 20 chars."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return []

    if not isinstance(data, dict):
        return []

    results: list[DiscoveredPrompt] = []
    _walk_dict(data, path, results)
    return results


def _walk_dict(data: dict, path: Path, results: list, depth: int = 0):
    """Recursively walk dict looking for prompt keys."""
    if depth > 10:  # Prevent infinite recursion
        return
    for key, value in data.items():
        if isinstance(value, str) and key.lower() in PROMPT_KEYS and len(value) > 20:
            results.append(DiscoveredPrompt(
                file=str(path),
                line=1,  # YAML doesn't easily give line numbers
                framework="config",
                origin="file",
                prompt_text=value,
                origin_file=str(path),
                estimated_tokens=len(value.split()),
                confidence=0.5,
            ))
        elif isinstance(value, dict):
            _walk_dict(value, path, results, depth + 1)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _walk_dict(item, path, results, depth + 1)


def _scan_template_dir(dir_path: Path) -> list[DiscoveredPrompt]:
    """Scan directory for template files."""
    results = []
    for template_file in dir_path.iterdir():
        if not template_file.is_file():
            continue
        if template_file.suffix not in TEMPLATE_EXTENSIONS:
            continue
        try:
            content = template_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if len(content.strip()) < 20:
            continue
        results.append(DiscoveredPrompt(
            file=str(template_file),
            line=1,
            framework="template",
            origin="file",
            prompt_text=content.strip(),
            origin_file=str(template_file),
            estimated_tokens=len(content.split()),
            confidence=0.6,
        ))
    return results
