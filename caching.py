import hashlib
import json
from typing import Dict, List, Optional

from redis import Redis

redis = Redis(decode_responses=True, ssl=False, health_check_interval=10)


def _is_redis_available() -> bool:
    try:
        assert redis and redis.ping()
        return True
    except Exception:
        return False


def _hash_sha256(s: str) -> str:
    hasher = hashlib.new("sha256")
    hasher.update(s.encode())
    return hasher.hexdigest()


def _serialize_request(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    timeout: int,
    max_output_tokens: Optional[int],
) -> str:
    return (
        json.dumps(messages)
        + model
        + str(temperature)
        + str(timeout)
        + str(max_output_tokens)
    )


def set_cache_response(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    timeout: int,
    max_output_tokens: Optional[int],
    response: str,
):
    if not _is_redis_available():
        return
    input_str = _serialize_request(
        messages,
        model,
        temperature,
        timeout,
        max_output_tokens,
    )
    assert redis
    redis.set(_hash_sha256(input_str), response)


def get_cached_response(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    timeout: int,
    max_output_tokens: Optional[int],
) -> Optional[str]:
    if not _is_redis_available():
        return None
    input_str = _serialize_request(
        messages,
        model,
        temperature,
        timeout,
        max_output_tokens,
    )
    cached_value = redis.get(_hash_sha256(input_str))
    if cached_value is None:
        return None
    return cached_value
