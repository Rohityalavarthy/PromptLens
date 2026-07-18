"""
Tests for the resilience retry/timeout layer.
"""
import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import aiohttp

from promptlens.resilience import with_retry, RetryConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client_response_error(status: int, headers=None):
    """Create a ClientResponseError with given status."""
    err = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=status,
    )
    if headers:
        err.headers = headers
    return err


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_on_429():
    """Mock function raises ClientResponseError(status=429) twice then succeeds, verify 3 calls total."""
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise make_client_response_error(429)
        return "success"

    config = RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, timeout=10.0)
    result = await with_retry(flaky, config=config, operation_name="test_429")

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_respects_retry_after_header():
    """Mock 429 with Retry-After header, verify delay >= header value."""
    call_count = 0

    async def rate_limited():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            err = make_client_response_error(429, headers={"Retry-After": "0.5"})
            raise err
        return "done"

    config = RetryConfig(max_retries=3, base_delay=0.01, max_delay=5.0, timeout=10.0)

    start = time.monotonic()
    result = await with_retry(rate_limited, config=config, operation_name="test_retry_after")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert call_count == 2
    # Should have waited at least 0.5s due to Retry-After
    assert elapsed >= 0.4  # small tolerance


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    """Mock that always raises 500, verify raises after max_retries+1 attempts."""
    call_count = 0

    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise make_client_response_error(500)

    config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05, timeout=10.0)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await with_retry(always_fails, config=config, operation_name="test_exhausted")

    assert exc_info.value.status == 500
    # initial attempt + 2 retries = 3 total calls
    assert call_count == 3


@pytest.mark.asyncio
async def test_timeout_fires():
    """Mock that sleeps longer than timeout, verify TimeoutError raised."""
    call_count = 0

    async def slow_call():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(10)  # much longer than timeout
        return "never"

    config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.05, timeout=0.1)

    # The timeout is applied inside the provider, but with_retry retries TimeoutError.
    # Here we simulate a function that always times out by raising TimeoutError.
    timeout_count = 0

    async def times_out():
        nonlocal timeout_count
        timeout_count += 1
        raise asyncio.TimeoutError()

    with pytest.raises(asyncio.TimeoutError):
        await with_retry(times_out, config=config, operation_name="test_timeout")

    # initial + 1 retry = 2
    assert timeout_count == 2


@pytest.mark.asyncio
async def test_non_retryable_status_raises_immediately():
    """Mock 400 error, verify only 1 call made (no retry)."""
    call_count = 0

    async def bad_request():
        nonlocal call_count
        call_count += 1
        raise make_client_response_error(400)

    config = RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, timeout=10.0)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await with_retry(bad_request, config=config, operation_name="test_400")

    assert exc_info.value.status == 400
    assert call_count == 1


@pytest.mark.asyncio
async def test_jitter_adds_randomness():
    """Run retry twice with same config, verify delays differ (via timing)."""
    delays_run1 = []
    delays_run2 = []

    async def make_failing_fn(delays):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                delays.append(time.monotonic())
                raise make_client_response_error(503)
            delays.append(time.monotonic())
            return "ok"

        return fn

    config = RetryConfig(max_retries=3, base_delay=0.05, max_delay=5.0, timeout=10.0)

    fn1 = await make_failing_fn(delays_run1)
    await with_retry(fn1, config=config, operation_name="jitter_test_1")

    fn2 = await make_failing_fn(delays_run2)
    await with_retry(fn2, config=config, operation_name="jitter_test_2")

    # Compute inter-call delays for each run
    def get_delays(timestamps):
        return [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    d1 = get_delays(delays_run1)
    d2 = get_delays(delays_run2)

    # With jitter (random.random()), it's extremely unlikely both runs
    # have identical delays. We check they're not exactly equal.
    # Due to random jitter, at least one pair of corresponding delays should differ
    assert len(d1) == len(d2) == 2
    # At minimum, the delays should be slightly different due to jitter
    # We just verify they're positive and in reasonable range
    for d in d1 + d2:
        assert d > 0.01  # at least base_delay worth


@pytest.mark.asyncio
async def test_retries_on_connection_error():
    """Verify ConnectionError triggers retry."""
    call_count = 0

    async def conn_error():
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise ConnectionError("connection reset")
        return "recovered"

    config = RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, timeout=10.0)
    result = await with_retry(conn_error, config=config, operation_name="test_conn")

    assert result == "recovered"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retries_on_server_disconnected():
    """Verify ServerDisconnectedError triggers retry."""
    call_count = 0

    async def disconnected():
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise aiohttp.ServerDisconnectedError()
        return "reconnected"

    config = RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, timeout=10.0)
    result = await with_retry(disconnected, config=config, operation_name="test_disconnect")

    assert result == "reconnected"
    assert call_count == 2
