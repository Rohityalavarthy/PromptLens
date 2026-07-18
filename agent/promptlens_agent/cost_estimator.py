from dataclasses import dataclass

# Approximate costs per 1M tokens (Together AI default pricing)
COST_PER_M_INPUT = 0.60   # Llama 3.3 70B Turbo
COST_PER_M_OUTPUT = 0.60


@dataclass
class CostEstimate:
    api_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    estimated_time_seconds: float  # based on ~2 calls/sec throughput


def estimate_shapley_cost(
    token_count: int, n_phrases: int, m_samples: int, n_test_inputs: int
) -> CostEstimate:
    """
    Estimate API calls and cost for a Shapley analysis run.

    Each walk: n_phrases generate calls (one per phrase subset).
    Total walks = m_samples * n_test_inputs.
    Each call: ~token_count input tokens, ~200 output tokens.
    """
    total_walks = m_samples * n_test_inputs
    generate_calls = total_walks * n_phrases
    input_tokens = generate_calls * token_count
    output_tokens = generate_calls * 200
    cost = (input_tokens * COST_PER_M_INPUT + output_tokens * COST_PER_M_OUTPUT) / 1_000_000
    time_seconds = generate_calls / 2.0  # ~2 calls/sec with concurrency
    return CostEstimate(
        api_calls=generate_calls,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=round(cost, 4),
        estimated_time_seconds=round(time_seconds, 1),
    )


def estimate_compression_cost(
    token_count: int, n_phrases: int, n_test_inputs: int
) -> CostEstimate:
    """
    Estimate cost for compression (1 LLM call) + validation (2 * n_test_inputs calls).
    """
    compression_calls = 1
    validation_calls = 2 * n_test_inputs  # original + compressed per input
    total_calls = compression_calls + validation_calls
    input_tokens = total_calls * token_count
    output_tokens = total_calls * 200
    cost = (input_tokens * COST_PER_M_INPUT + output_tokens * COST_PER_M_OUTPUT) / 1_000_000
    time_seconds = total_calls / 2.0
    return CostEstimate(
        api_calls=total_calls,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=round(cost, 4),
        estimated_time_seconds=round(time_seconds, 1),
    )


def format_estimate(estimate: CostEstimate) -> str:
    """Format a cost estimate for terminal display."""
    return (
        f"  API calls: {estimate.api_calls:,}\n"
        f"  Input tokens: ~{estimate.estimated_input_tokens:,}\n"
        f"  Output tokens: ~{estimate.estimated_output_tokens:,}\n"
        f"  Estimated cost: ${estimate.estimated_cost_usd:.4f}\n"
        f"  Estimated time: {estimate.estimated_time_seconds:.0f}s"
    )


def estimate_to_dict(estimate: CostEstimate) -> dict:
    """Convert estimate to dict for JSON output."""
    return {
        "api_calls": estimate.api_calls,
        "estimated_input_tokens": estimate.estimated_input_tokens,
        "estimated_output_tokens": estimate.estimated_output_tokens,
        "estimated_cost_usd": estimate.estimated_cost_usd,
        "estimated_time_seconds": estimate.estimated_time_seconds,
    }
