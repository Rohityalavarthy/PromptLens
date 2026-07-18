import re
from pathlib import Path
from .discovery import DiscoveredPrompt

TS_PATTERNS = [
    # OpenAI messages array with system role (single-line, double/single quotes)
    (r'role:\s*["\']system["\'].*?content:\s*["\'](.+?)["\']', "literal"),
    # Template literal content for system role
    (r'role:\s*["\']system["\'].*?content:\s*`([^`]+)`', "literal"),
    # Anthropic system kwarg (quotes)
    (r'system:\s*["\'](.+?)["\']', "literal"),
    # Anthropic system kwarg (template literal)
    (r'system:\s*`([^`]+)`', "literal"),
    # Prompt variable declarations (quotes)
    (r'(?:const|let|var)\s+\w*(?:[Pp]rompt|PROMPT|[Ss]ystem|SYSTEM)\w*\s*=\s*["\'](.+?)["\']', "variable"),
    # Prompt variable declarations (template literal)
    (r'(?:const|let|var)\s+\w*(?:[Pp]rompt|PROMPT|[Ss]ystem|SYSTEM)\w*\s*=\s*`([^`]+)`', "variable"),
]

TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".mts"}
SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", "dist", "build", ".next"}


def discover_prompts_ts(repo_path: str) -> list[DiscoveredPrompt]:
    """Scan TypeScript/JavaScript files for LLM prompt patterns."""
    discovered = []
    repo = Path(repo_path)

    for ts_file in repo.rglob("*"):
        if ts_file.suffix not in TS_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in ts_file.parts):
            continue
        if not ts_file.is_file():
            continue
        try:
            source = ts_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        for pattern, origin in TS_PATTERNS:
            for match in re.finditer(pattern, source, re.DOTALL):
                text = match.group(1).strip()
                if len(text) < 20:  # Skip trivially short matches
                    continue
                line_no = source[:match.start()].count('\n') + 1
                discovered.append(DiscoveredPrompt(
                    file=str(ts_file),
                    line=line_no,
                    framework="typescript",
                    origin=origin,
                    prompt_text=text,
                    origin_file=None,
                    estimated_tokens=len(text.split()),
                    confidence=0.7,
                ))
    return discovered
