import pytest
from pathlib import Path

from promptlens_agent.discovery_config import discover_prompts_config


def test_discovers_system_prompt_yaml_key(tmp_path: Path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        'system_prompt: "You are a helpful assistant that provides accurate answers to questions."\n'
    )
    results = discover_prompts_config(str(tmp_path))
    assert len(results) >= 1
    assert any("helpful assistant" in (r.prompt_text or "") for r in results)
    assert results[0].framework == "config"
    assert results[0].origin == "file"


def test_discovers_nested_prompt_key(tmp_path: Path):
    yaml_file = tmp_path / "nested.yml"
    yaml_file.write_text(
        'llm:\n'
        '  model: gpt-4\n'
        '  prompt: "You are a customer support agent that helps users resolve their issues."\n'
    )
    results = discover_prompts_config(str(tmp_path))
    assert len(results) >= 1
    assert any("customer support" in (r.prompt_text or "") for r in results)


def test_ignores_short_values(tmp_path: Path):
    yaml_file = tmp_path / "short.yaml"
    yaml_file.write_text(
        'system_prompt: "Be helpful"\n'
    )
    results = discover_prompts_config(str(tmp_path))
    assert len(results) == 0


def test_discovers_template_file_in_prompts_dir(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template_file = prompts_dir / "system.jinja2"
    template_file.write_text(
        "You are a helpful assistant. Answer the user's question based on the following context:\n"
        "{{ context }}\n"
    )
    results = discover_prompts_config(str(tmp_path))
    assert len(results) >= 1
    assert any(r.framework == "template" for r in results)
    assert any("helpful assistant" in (r.prompt_text or "") for r in results)


def test_skips_invalid_yaml(tmp_path: Path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(
        ':\n  - invalid: [yaml\n  unclosed\n'
    )
    # Should not raise an exception
    results = discover_prompts_config(str(tmp_path))
    assert isinstance(results, list)
