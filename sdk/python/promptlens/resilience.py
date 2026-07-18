"""
Retry and timeout layer for PromptLens API calls.

Provides exponential backoff with jitter for transient failures,
timeout protection, and structured logging.
"""

import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import TypeVar, Callable, Awaitable

import aiohttp

logger = logging.getLogger("promptlens.resilience")

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    timeout: float = 60.0
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


DEFAULT_CONFIG = RetryConfig()


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig = DEFAULT_CONFIG,
    operation_name: str = "api_call",
) -> T:
    """
    Execute fn with exponential backoff retry on transient failures.

    Retries on:
    - aiohttp.ClientResponseError with status in retryable_status_codes
    - asyncio.TimeoutError
    - aiohttp.ServerDisconnectedError
    - ConnectionError

    Backoff: delay = min(base_delay * 2^attempt + random(0, 1), max_delay)
    Respects Retry-After header for 429 responses.
    Logs each retry with operation_name, attempt number, and error.
    Raises the last exception after exhausting retries.
    """
    last_exception: BaseException | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except aiohttp.ClientResponseError as e:
            last_exception = e
            if e.status not in config.retryable_status_codes:
                raise
            if attempt >= config.max_retries:
                logger.error(
                    f"Exhausted {config.max_retries} retries for {operation_name}: "
                    f"{type(e).__name__} - {e}"
                )
                raise

            # Check for Retry-After header on 429
            delay = min(config.base_delay * (2 ** attempt) + random.random(), config.max_delay)
            if e.status == 429 and hasattr(e, "headers") and e.headers:
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    try:
                        retry_after_secs = float(retry_after)
                        logger.info(
                            f"Rate limited on {operation_name}, waiting {retry_after_secs}s (Retry-After header)"
                        )
                        delay = max(delay, retry_after_secs)
                    except (ValueError, TypeError):
                        pass

            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {operation_name}: "
                f"{type(e).__name__} - {e}"
            )
            await asyncio.sleep(delay)

        except (asyncio.TimeoutError, aiohttp.ServerDisconnectedError, ConnectionError) as e:
            last_exception = e
            if attempt >= config.max_retries:
                logger.error(
                    f"Exhausted {config.max_retries} retries for {operation_name}: "
                    f"{type(e).__name__} - {e}"
                )
                raise

            delay = min(config.base_delay * (2 ** attempt) + random.random(), config.max_delay)
            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {operation_name}: "
                f"{type(e).__name__} - {e}"
            )
            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    raise last_exception  # type: ignore[misc]
