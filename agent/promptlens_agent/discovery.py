import ast
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiscoveredPrompt:
    file: str
    line: int
    framework: str
    origin: str             # "literal" | "variable" | "file" | "unknown"
    prompt_text: Optional[str]   # None if origin is "unknown" (e.g., DB-fetched)
    origin_file: Optional[str]   # if origin == "file"
    estimated_tokens: int


# API call signatures to detect — extend as needed
FRAMEWORK_SIGNATURES = {
    "openai":     ["openai.chat.completions.create", "client.chat.completions.create"],
    "anthropic":  ["client.messages.create", "anthropic.messages.create"],
    "langchain":  ["ChatOpenAI", "ChatAnthropic", "LLMChain", "PromptTemplate"],
    "bedrock":    ["bedrock_runtime.invoke_model", "BedrockChat"],
}


class PythonPromptVisitor(ast.NodeVisitor):
    """AST visitor that finds LLM API calls and extracts system prompt arguments."""

    def __init__(self, source_lines: list[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.discovered: list[DiscoveredPrompt] = []
        self._assignments: dict[str, str] = {}  # var_name -> literal value

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track simple string assignments for variable resolution."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._assignments[target.id] = node.value.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect LLM API calls and extract system prompt."""
        call_str = ast.unparse(node)

        for framework, signatures in FRAMEWORK_SIGNATURES.items():
            if any(sig in call_str for sig in signatures):
                prompt_text, origin, origin_file = self._extract_system_prompt(node)
                if prompt_text or origin != "unknown":
                    self.discovered.append(DiscoveredPrompt(
                        file=self.file_path,
                        line=node.lineno,
                        framework=framework,
                        origin=origin,
                        prompt_text=prompt_text,
                        origin_file=origin_file,
                        estimated_tokens=len(prompt_text.split()) if prompt_text else 0,
                    ))

        self.generic_visit(node)

    def _extract_system_prompt(self, node: ast.Call):
        """
        Look for system message in messages=[{"role": "system", "content": ...}].
        Returns (prompt_text, origin, origin_file).
        """
        for keyword in node.keywords:
            if keyword.arg == "messages" and isinstance(keyword.value, ast.List):
                for elt in keyword.value.elts:
                    if isinstance(elt, ast.Dict):
                        role_val = self._get_dict_value(elt, "role")
                        if role_val == "system":
                            content_val = self._get_dict_value_node(elt, "content")
                            return self._resolve_value(content_val)

        return None, "unknown", None

    def _get_dict_value(self, node: ast.Dict, key: str) -> Optional[str]:
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                if isinstance(v, ast.Constant):
                    return v.value
        return None

    def _get_dict_value_node(self, node: ast.Dict, key: str):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
        return None

    def _resolve_value(self, node):
        """Resolve a value node to (text, origin, origin_file)."""
        if node is None:
            return None, "unknown", None
        if isinstance(node, ast.Constant):
            return node.value, "literal", None
        if isinstance(node, ast.Name):
            if node.id in self._assignments:
                return self._assignments[node.id], "variable", None
            return None, "variable", None
        if isinstance(node, ast.Call):
            call_str = ast.unparse(node)
            # Detect open("path").read() or Path("path").read_text()
            file_match = re.search(r'["\']([^"\']+\.(txt|md|jinja2|j2))["\']', call_str)
            if file_match:
                return None, "file", file_match.group(1)
        return None, "unknown", None


def discover_prompts(repo_path: str) -> list[DiscoveredPrompt]:
    """
    Walk repo, find all Python files, run AST visitor on each.
    Returns flat list of discovered prompts across the codebase.
    """
    discovered = []
    repo = Path(repo_path)

    for py_file in repo.rglob("*.py"):
        # Skip venv, node_modules, __pycache__, test files
        skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
        if any(part in skip_dirs for part in py_file.parts):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = source.splitlines()
            visitor = PythonPromptVisitor(lines, str(py_file))
            visitor.visit(tree)

            # Resolve file-origin prompts
            for dp in visitor.discovered:
                if dp.origin == "file" and dp.origin_file:
                    prompt_path = repo / dp.origin_file
                    if prompt_path.exists():
                        dp.prompt_text = prompt_path.read_text(encoding="utf-8")
                        dp.estimated_tokens = len(dp.prompt_text.split())

            discovered.extend(visitor.discovered)
        except (SyntaxError, UnicodeDecodeError):
            continue  # skip unparseable files

    return discovered
