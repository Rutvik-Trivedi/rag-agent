from typing import List, Tuple
from functools import lru_cache

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
import torch

from llm import async_llm as openai_embedder

MAX_N_INPUT_TEXTS = 100
TOP_N_CHUNKS = 5

RERANKER_MODEL_CARD = "jinaai/jina-reranker-v1-turbo-en"
OPENAI_SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small"
]

@lru_cache
def load_reranker_model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_CARD, trust_remote_code=True)

@lru_cache
def load_embedding_model_and_tokenizer() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_CARD, trust_remote_code=True, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_CARD)
    return model, tokenizer

async def _embed_openai(
    text_list: List[str],
    model: str,
) -> np.ndarray:

    # Create groups to reduce OpenAI API calls
    # IMPROVEMENT: Create groups based on total number of input tokens instead of a hardcoded
    # group size
    groups = [
        text_list[i : i + MAX_N_INPUT_TEXTS]
        for i in range(0, len(text_list), MAX_N_INPUT_TEXTS)
    ]

    embeddings = []
    for group in groups:
        group_embeddings = await openai_embedder.embeddings.create(
            input=group, model=model
        )
        embeddings.extend(
            [np.array(embedding.embedding) for embedding in group_embeddings.data]
        )
    return np.array(embeddings)

def _embed_transformers(
    text_list: List[str],
) -> np.ndarray:
    model, tokenizer = load_embedding_model_and_tokenizer()
    tokenized_texts = tokenizer(
        text_list, truncation=True, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = model(**tokenized_texts, output_hidden_states=True).hidden_states[-1]
    weights_for_non_padding = tokenized_texts.attention_mask * torch.arange(start=1, end=embeddings.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(embeddings * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
    return sentence_embeddings.numpy()


async def embed(
    text_list: List[str],
    model: str,
) -> np.ndarray:
    if model in OPENAI_SUPPORTED_EMBEDDING_MODELS:
        return await _embed_openai(text_list, model)
    else:
        return _embed_transformers(text_list)

def _rerank(
    text_chunks: List[str],
    query: str,
    top_k: int
) -> List[str]:
    model = load_reranker_model()
    results = model.rank(query, text_chunks, top_k=top_k, return_documents=True)
    return [result["text"] for result in results]
    

def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


async def get_top_text_chunks(
    text_chunks: List[str],
    query: str,
    model: str,
    top_k: int = TOP_N_CHUNKS,
    rerank: bool = False,
    reranker_top_k: int = TOP_N_CHUNKS * 2
) -> List[str]:
    text_chunk_embeddings = await embed(text_chunks, model)
    query_embedding = await embed([query], model)
    query_embedding = query_embedding[0]

    similarities = [
        _cosine_similarity(query_embedding, text_chunk_embedding)
        for text_chunk_embedding in text_chunk_embeddings
    ]
    sorted_similarities_with_texts = sorted(
        list(zip(text_chunks, similarities)), reverse=True, key=lambda x: x[1]
    )
    top_texts = [text for text, _ in sorted_similarities_with_texts][:reranker_top_k]
    if rerank:
        top_texts = _rerank(top_texts, query, top_k)
    else:
        top_texts = top_texts[:top_k]
    return top_texts
