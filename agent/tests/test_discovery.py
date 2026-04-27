import tempfile
import textwrap
from pathlib import Path
from promptlens_agent.discovery import discover_prompts, PythonPromptVisitor
import ast


def make_temp_py(content: str) -> Path:
    """Write content to a temp .py file, return its path."""
    d = tempfile.mkdtemp()
    f = Path(d) / "test_app.py"
    f.write_text(textwrap.dedent(content))
    return f


def test_discovers_literal_system_prompt():
    src = """
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].framework == "openai"
    assert results[0].origin == "literal"
    assert results[0].prompt_text == "You are a helpful assistant."


def test_discovers_variable_system_prompt():
    src = """
    import openai
    SYSTEM_PROMPT = "You are a customer support agent."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].origin == "variable"
    assert results[0].prompt_text == "You are a customer support agent."


def test_skips_calls_without_system_message():
    src = """
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 0


def test_multiple_calls_discovered():
    src = """
    import openai
    r1 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Prompt one."}, {"role": "user", "content": q}]
    )
    r2 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Prompt two."}, {"role": "user", "content": q}]
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 2
    prompts = {r.prompt_text for r in results}
    assert "Prompt one." in prompts
    assert "Prompt two." in prompts


def test_estimated_tokens_counts_words():
    src = """
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful and friendly assistant."},
            {"role": "user", "content": q},
        ]
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert results[0].estimated_tokens == 7  # word count of "You are a helpful and friendly assistant."


def test_syntax_error_file_is_skipped(tmp_path):
    bad_py = tmp_path / "bad.py"
    bad_py.write_text("def broken(\n    this is not python")
    results = discover_prompts(str(tmp_path))
    assert results == []
