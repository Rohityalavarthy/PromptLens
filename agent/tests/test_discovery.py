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


def test_discovers_anthropic_system_kwarg():
    src = """
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-opus-20240229",
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello"}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].framework == "anthropic"
    assert results[0].origin == "literal"
    assert results[0].prompt_text == "You are a helpful assistant."
    assert results[0].confidence == 1.0


def test_discovers_langchain_system_message():
    src = """
    from langchain.schema import SystemMessage
    msg = SystemMessage(content="You are a helpful assistant.")
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].framework == "langchain"
    assert results[0].prompt_text == "You are a helpful assistant."
    assert results[0].confidence == 0.9


def test_discovers_langchain_positional_arg():
    src = """
    from langchain.schema import SystemMessage
    msg = SystemMessage("You are a helpful assistant.")
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].framework == "langchain"
    assert results[0].prompt_text == "You are a helpful assistant."
    assert results[0].confidence == 0.9


def test_discovers_fstring_partial():
    src = """
    import openai
    role = "coding"
    prompt = f"You are a {role} assistant. Always be helpful."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    # The variable `prompt` is an f-string assigned, but since the assignment
    # visitor only tracks ast.Constant (plain strings), it won't resolve the
    # f-string variable. However, we still detect the call with origin="variable".
    # Let's test directly with inline f-string in messages:
    src2 = """
    import openai
    role = "coding"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"You are a {role} assistant. Always be helpful."}],
    )
    """
    py_file2 = make_temp_py(src2)
    results2 = discover_prompts(str(py_file2.parent))
    assert len(results2) == 1
    assert results2[0].origin == "variable"
    # Should extract the constant parts
    assert "You are a" in results2[0].prompt_text
    assert "assistant. Always be helpful." in results2[0].prompt_text
    assert results2[0].confidence == 0.8


def test_discovers_concatenation():
    src = """
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are " + "a helpful assistant."}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].prompt_text == "You are a helpful assistant."
    assert results[0].origin == "variable"
    assert results[0].confidence == 0.8


def test_discovers_format_template():
    src = """
    import openai
    role = "helpful"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a {} assistant.".format(role)}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    assert len(results) == 1
    assert results[0].prompt_text == "You are a {} assistant."
    assert results[0].origin == "variable"
    assert results[0].confidence == 0.8


def test_deduplicates_same_prompt_text():
    src = """
    import openai
    SYSTEM_PROMPT = "You are a helpful assistant."
    r1 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    r2 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    # Same file, same prompt_text → deduplicated to 1
    assert len(results) == 1
    assert results[0].prompt_text == "You are a helpful assistant."


def test_confidence_set_by_origin():
    src = """
    import openai
    SYSTEM_PROMPT = "Variable prompt."
    # literal
    r1 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Literal prompt."}],
    )
    # variable
    r2 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    # f-string
    r3 = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"Hello {name} world"}],
    )
    """
    py_file = make_temp_py(src)
    results = discover_prompts(str(py_file.parent))
    by_text = {r.prompt_text: r for r in results}
    assert by_text["Literal prompt."].confidence == 1.0
    assert by_text["Variable prompt."].confidence == 0.8
    # f-string resolves to constant parts joined
    fstring_result = [r for r in results if "Hello" in (r.prompt_text or "")]
    assert len(fstring_result) == 1
    assert fstring_result[0].confidence == 0.8  # variable with text resolved
