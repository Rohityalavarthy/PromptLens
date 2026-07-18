import pytest
from pathlib import Path

from promptlens_agent.discovery_ts import discover_prompts_ts


def test_discovers_openai_system_message_ts(tmp_path: Path):
    ts_file = tmp_path / "app.ts"
    ts_file.write_text(
        'const response = await client.chat.completions.create({\n'
        '  messages: [{ role: "system", content: "You are a helpful coding assistant." }]\n'
        '});\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) == 1
    assert results[0].prompt_text == "You are a helpful coding assistant."
    assert results[0].framework == "typescript"
    assert results[0].origin == "literal"


def test_discovers_template_literal_prompt(tmp_path: Path):
    ts_file = tmp_path / "chat.ts"
    ts_file.write_text(
        'const msg = { role: "system", content: `You are a helpful assistant that helps users with coding tasks and questions.` };\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) == 1
    assert "helpful assistant" in results[0].prompt_text
    assert results[0].origin == "literal"


def test_discovers_anthropic_system_ts(tmp_path: Path):
    ts_file = tmp_path / "anthropic.ts"
    ts_file.write_text(
        'const response = await client.messages.create({\n'
        '  system: "You are a helpful assistant that answers questions clearly.",\n'
        '  messages: [{ role: "user", content: "Hello" }]\n'
        '});\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) >= 1
    system_results = [r for r in results if "helpful assistant" in (r.prompt_text or "")]
    assert len(system_results) >= 1


def test_discovers_prompt_variable_ts(tmp_path: Path):
    ts_file = tmp_path / "config.ts"
    ts_file.write_text(
        'const SYSTEM_PROMPT = "You are a helpful assistant that provides accurate information to users.";\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) == 1
    assert results[0].origin == "variable"
    assert "helpful assistant" in results[0].prompt_text


def test_skips_short_matches(tmp_path: Path):
    ts_file = tmp_path / "short.ts"
    ts_file.write_text(
        'const SYSTEM_PROMPT = "Be helpful";\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) == 0


def test_skips_node_modules(tmp_path: Path):
    nm_dir = tmp_path / "node_modules" / "some-pkg"
    nm_dir.mkdir(parents=True)
    ts_file = nm_dir / "index.ts"
    ts_file.write_text(
        'const SYSTEM_PROMPT = "You are a helpful assistant that provides accurate information to users.";\n'
    )
    results = discover_prompts_ts(str(tmp_path))
    assert len(results) == 0
