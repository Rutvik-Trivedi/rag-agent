from typing import List

import pymupdf4llm
from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
MARKDOWN_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON
)


def _read_pdf(pdf_file_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_file_path)


def _chunk_pdf(markdown_text: str, splitter: str) -> List[str]:
    if splitter == "recursive":
        documents = RECURSIVE_SPLITTER.create_documents([markdown_text])
    elif splitter == "markdown_header":
        documents = MARKDOWN_HEADER_SPLITTER.split_text(markdown_text)
    else:
        raise ValueError(
            f"{splitter} splitter not supported. Splitter must be 'recursive' or 'markdown_header'"
        )
    return [document.page_content for document in documents]


def get_pdf_chunks(pdf_file_path: str, splitter: str = "recursive") -> List[str]:
    markdown_text = _read_pdf(pdf_file_path)
    return _chunk_pdf(markdown_text, splitter)
