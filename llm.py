import os
from time import sleep
from typing import *

from openai import (APIError, APITimeoutError, AsyncOpenAI, AsyncStream,
                    RateLimitError)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from caching import get_cached_response, set_cache_response
from prompts import QUERY_PROMPT, SYSTEM_MESSAGE

async_llm = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)
MAX_API_ERROR_COUNT_BEFORE_RAISING = 5


def _parse_completion_result(
    completion: ChatCompletion,
) -> str:
    choice = [choice.model_dump() for choice in completion.choices][0]
    message = choice["message"]
    # Code does not support parsing of openai tool calls
    # IMPROVEMENT: Not required for the scope of this usecase, but support tool calls
    return message["content"]


async def async_complete(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: Optional[int] = None,
    timeout: int = 180,
) -> str:
    cached_completion = get_cached_response(
        messages, model, temperature, timeout, max_output_tokens
    )
    if cached_completion:
        return cached_completion
    api_error_count = 0
    completions_create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    while True:
        try:
            completion: Union[ChatCompletion, AsyncStream[ChatCompletionChunk]] = (
                await async_llm.chat.completions.create(**completions_create_kwargs)
            )
            break
        except RateLimitError as e:
            if e.code == "insufficient_quota":
                raise
            print(f"Sleeping due to {e}")
            sleep(5)
        except APITimeoutError:
            # OpenAI recently added a caching feature, which should fix the API
            # Timeout on the next run
            pass
        except APIError as e:
            print(f"Sleeping due to {e}")
            sleep(10)
            api_error_count += 1
            if api_error_count == MAX_API_ERROR_COUNT_BEFORE_RAISING:
                # Do not sleep infinitely
                raise

    assert isinstance(completion, ChatCompletion)
    completion_result = _parse_completion_result(completion)
    set_cache_response(
        messages, model, temperature, timeout, max_output_tokens, completion_result
    )
    return completion_result


def create_messages(query: str, text_chunks: List[str]) -> List[Dict[str, str]]:
    context_info = "\n".join(text_chunks)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": QUERY_PROMPT.format(context=context_info, question=query),
        },
    ]
