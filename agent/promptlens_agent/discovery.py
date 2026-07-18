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
    confidence: float = 1.0  # 1.0=literal, 0.9=langchain class, 0.8=variable, 0.5=f-string partial


# API call signatures to detect — extend as needed
FRAMEWORK_SIGNATURES = {
    "openai":     ["openai.chat.completions.create", "client.chat.completions.create"],
    "anthropic":  ["client.messages.create", "anthropic.messages.create"],
    "langchain":  ["ChatOpenAI", "ChatAnthropic", "LLMChain", "PromptTemplate"],
    "bedrock":    ["bedrock_runtime.invoke_model", "BedrockChat"],
}


class PythonPromptVisitor(ast.NodeVisitor):
    """AST visitor that finds LLM API calls and extracts system prompt arguments."""

    LANGCHAIN_MESSAGE_CLASSES = {
        "SystemMessage", "HumanMessage", "AIMessage",
        "SystemMessagePromptTemplate", "HumanMessagePromptTemplate",
    }

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
                    confidence = self._confidence_for_origin(origin, prompt_text)
                    self.discovered.append(DiscoveredPrompt(
                        file=self.file_path,
                        line=node.lineno,
                        framework=framework,
                        origin=origin,
                        prompt_text=prompt_text,
                        origin_file=origin_file,
                        estimated_tokens=len(prompt_text.split()) if prompt_text else 0,
                        confidence=confidence,
                    ))

        if self._is_langchain_message_class(node):
            prompt_text, origin, origin_file = self._extract_langchain_content(node)
            if prompt_text:
                self.discovered.append(DiscoveredPrompt(
                    file=self.file_path,
                    line=node.lineno,
                    framework="langchain",
                    origin=origin,
                    prompt_text=prompt_text,
                    origin_file=origin_file,
                    estimated_tokens=len(prompt_text.split()) if prompt_text else 0,
                    confidence=0.9,
                ))

        self.generic_visit(node)

    def _extract_system_prompt(self, node: ast.Call):
        """
        Look for system message in messages=[{"role": "system", "content": ...}]
        or system= kwarg (Anthropic pattern).
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

        for keyword in node.keywords:
            if keyword.arg == "system":
                return self._resolve_value(keyword.value)

        return None, "unknown", None

    def _is_langchain_message_class(self, node: ast.Call) -> bool:
        """Check if a call node is a LangChain message class instantiation."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.LANGCHAIN_MESSAGE_CLASSES
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in self.LANGCHAIN_MESSAGE_CLASSES
        return False

    def _extract_langchain_content(self, node: ast.Call) -> tuple:
        """Extract the content argument from a LangChain message class call."""
        for keyword in node.keywords:
            if keyword.arg == "content":
                return self._resolve_value(keyword.value)
        if node.args:
            return self._resolve_value(node.args[0])
        return None, "unknown", None

    @staticmethod
    def _confidence_for_origin(origin: str, prompt_text: Optional[str]) -> float:
        """Return confidence score based on the origin type."""
        if origin == "literal":
            return 1.0
        if origin == "variable":
            return 0.8 if prompt_text else 0.5
        return 1.0

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
        if isinstance(node, ast.JoinedStr):
            parts = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            if parts:
                return " ".join(parts), "variable", None
            return None, "unknown", None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left_text, _, _ = self._resolve_value(node.left)
            right_text, _, _ = self._resolve_value(node.right)
            if left_text and right_text:
                return left_text + right_text, "variable", None
            if left_text:
                return left_text, "variable", None
            return None, "unknown", None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "format" and isinstance(node.func.value, ast.Constant):
                    return node.func.value.value, "variable", None
            call_str = ast.unparse(node)
            # Detect open("path").read() or Path("path").read_text()
            file_match = re.search(r'["\']([^"\']+\.(txt|md|jinja2|j2))["\']', call_str)
            if file_match:
                return None, "file", file_match.group(1)
        return None, "unknown", None


def _process_file(py_file: Path, repo_root: Path) -> list[DiscoveredPrompt]:
    """Parse one Python file and return all discovered prompts."""
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        visitor = PythonPromptVisitor(lines, str(py_file))
        visitor.visit(tree)

        for dp in visitor.discovered:
            if dp.origin == "file" and dp.origin_file:
                prompt_path = repo_root / dp.origin_file
                if prompt_path.exists():
                    dp.prompt_text = prompt_path.read_text(encoding="utf-8")
                    dp.estimated_tokens = len(dp.prompt_text.split())

        return visitor.discovered
    except (SyntaxError, UnicodeDecodeError):
        return []


def discover_prompts_in_file(file_path: str) -> list[DiscoveredPrompt]:
    """
    Discover all prompts in a single Python file.
    repo_root is treated as the file's parent directory for resolving file-origin paths.
    """
    path = Path(file_path)
    if not path.exists() or path.suffix != ".py":
        return []
    return _process_file(path, path.parent)


def discover_prompts(repo_path: str) -> list[DiscoveredPrompt]:
    """
    Walk repo, find all Python files, run AST visitor on each.
    Returns flat list of discovered prompts across the codebase.
    """
    discovered = []
    repo = Path(repo_path)
    skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}

    for py_file in repo.rglob("*.py"):
        if any(part in skip_dirs for part in py_file.parts):
            continue
        discovered.extend(_process_file(py_file, repo))

    from .discovery_ts import discover_prompts_ts
    from .discovery_config import discover_prompts_config

    discovered.extend(discover_prompts_ts(repo_path))
    discovered.extend(discover_prompts_config(repo_path))

    seen: set[tuple[str, str | None]] = set()
    unique = []
    for dp in discovered:
        key = (dp.file, dp.prompt_text)
        if key not in seen:
            seen.add(key)
            unique.append(dp)
    return unique
