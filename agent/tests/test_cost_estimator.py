"""Tests for cost_estimator module."""
from promptlens_agent.cost_estimator import (
    estimate_shapley_cost,
    estimate_compression_cost,
    format_estimate,
    estimate_to_dict,
    CostEstimate,
)


def test_shapley_cost_scales_with_m_samples():
    """Doubling M roughly doubles api_calls and cost."""
    est1 = estimate_shapley_cost(token_count=150, n_phrases=8, m_samples=10, n_test_inputs=3)
    est2 = estimate_shapley_cost(token_count=150, n_phrases=8, m_samples=20, n_test_inputs=3)

    assert est2.api_calls == est1.api_calls * 2
    # Cost should roughly double (same formula, doubled calls)
    assert abs(est2.estimated_cost_usd / est1.estimated_cost_usd - 2.0) < 0.01


def test_cost_returns_positive_values():
    """No zeros or negatives for non-trivial inputs."""
    est = estimate_shapley_cost(token_count=100, n_phrases=5, m_samples=10, n_test_inputs=3)

    assert est.api_calls > 0
    assert est.estimated_input_tokens > 0
    assert est.estimated_output_tokens > 0
    assert est.estimated_cost_usd > 0
    assert est.estimated_time_seconds > 0


def test_compression_cost_scales_with_test_inputs():
    """More test inputs = more cost."""
    est1 = estimate_compression_cost(token_count=150, n_phrases=8, n_test_inputs=3)
    est2 = estimate_compression_cost(token_count=150, n_phrases=8, n_test_inputs=6)

    # More test inputs means more validation calls
    assert est2.api_calls > est1.api_calls
    assert est2.estimated_cost_usd > est1.estimated_cost_usd


def test_format_estimate_readable():
    """format_estimate returns multi-line string with expected fields."""
    est = CostEstimate(
        api_calls=100,
        estimated_input_tokens=15000,
        estimated_output_tokens=20000,
        estimated_cost_usd=0.0210,
        estimated_time_seconds=50.0,
    )
    output = format_estimate(est)

    assert "API calls: 100" in output
    assert "Input tokens:" in output
    assert "Output tokens:" in output
    assert "Estimated cost: $" in output
    assert "Estimated time:" in output
    # Should be multi-line
    assert output.count("\n") >= 4


def test_estimate_to_dict_has_all_fields():
    """Dict has all 5 expected keys."""
    est = CostEstimate(
        api_calls=50,
        estimated_input_tokens=7500,
        estimated_output_tokens=10000,
        estimated_cost_usd=0.0105,
        estimated_time_seconds=25.0,
    )
    d = estimate_to_dict(est)

    expected_keys = {
        "api_calls",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "estimated_cost_usd",
        "estimated_time_seconds",
    }
    assert set(d.keys()) == expected_keys
    assert d["api_calls"] == 50
    assert d["estimated_input_tokens"] == 7500
    assert d["estimated_output_tokens"] == 10000
    assert d["estimated_cost_usd"] == 0.0105
    assert d["estimated_time_seconds"] == 25.0
