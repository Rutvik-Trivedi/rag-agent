import asyncio
import datetime
import json
import os
from typing import Dict, List
from tqdm import tqdm

from dotenv import load_dotenv

from embedding import get_top_text_chunks
from llm import async_complete, create_messages
from pdf_processing import get_pdf_chunks
from slack import create_message_for_slack, send_message

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 2048
SEND_TO_SLACK = False
SLACK_CHANNEL = ""
RERANK = False  # set this to True for enabling reranking. CAUTION: Will require more memory
load_dotenv()


def _load_queries(queries_txt_path: str) -> List[str]:
    with open(queries_txt_path, "r", encoding="utf-8") as f:
        queries = f.read().split("\n")
    return [query for query in queries if query]


async def main(pdf_file_path: str, queries_txt_path: str) -> Dict[str, str]:
    pdf_chunks = get_pdf_chunks(pdf_file_path)

    queries = _load_queries(queries_txt_path)
    result_json: Dict[str, str] = {}
    for query in tqdm(queries, desc="Processing Queries"):
        top_text_chunks = await get_top_text_chunks(
            pdf_chunks, query, EMBEDDING_MODEL, rerank = RERANK
        )
        result_json[query] = await async_complete(
            create_messages(query, top_text_chunks),
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    return result_json


if __name__ == "__main__":
    pdf_file_path = "sample/handbook.pdf"
    queries_txt_path = "sample/input_questions.txt"
    result_json = asyncio.run(main(pdf_file_path, queries_txt_path))
    if not os.path.exists("output"):
        os.mkdir("output")
    run_log_time = int(datetime.datetime.now().timestamp())
    with open(f"output/output_{run_log_time}.json", "w") as f:
        json.dump(result_json, f, indent=4, sort_keys=True)
    if SEND_TO_SLACK:
        assert SLACK_CHANNEL
        message = create_message_for_slack(
            f"output/output_{run_log_time}.json", pdf_file_path
        )
        if send_message(message, SLACK_CHANNEL):
            print(f"Message sent to slack")
        else:
            print("Failed to send message to slack")