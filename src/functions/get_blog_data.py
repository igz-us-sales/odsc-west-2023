import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterable, List

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from tqdm import tqdm

from src.config import AppConfig, get_vector_store


logger = logging.getLogger(__name__)

config = AppConfig()


def apply_function_on_data_in_parallel(
    function: Callable, data: Iterable, description: str, **kwargs
) -> List[Any]:
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(data), desc=description, ncols=80) as pbar:
            for _, docs in enumerate(
                pool.imap_unordered(partial(function, **kwargs), data)
            ):
                results.append(docs)
                pbar.update()
    return results


def get_existing_milvus_documents(store: Milvus) -> set:
    if store.col:
        resp = store.col.query(expr="pk >= 0", output_fields=["source"])
        return {s["source"] for s in resp}
    else:
        return set()


def filter_urls(new_urls: List[str], existing_urls: List[str]) -> List[str]:
    return list(set(new_urls) - set(existing_urls))


def load_all_urls(urls: List[str]) -> List[Document]:
    loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"})
    return loader.load()


# def load_all_urls(urls: List[str]) -> List[Document]:
#     return apply_function_on_data_in_parallel(
#         function=load_single_url, data=urls, description="Loading new URLs"
#     )


def process_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    if not documents:
        logger.info("No new documents to load")
        return
    logger.info(f"Loaded {len(documents)} new documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logger.info(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)"
    )
    return texts


def ingest_urls(urls_file: str) -> None:

    # Load vector store
    store = get_vector_store(config=config)
    logger.info(f"Using vectorstore {type(store)}")

    # Get list of all URLs
    urls = Path(urls_file).read_text().splitlines()

    # Only ingest new URLs
    filtered_urls = filter_urls(
        new_urls=urls,
        existing_urls=get_existing_milvus_documents(store=store),
    )
    documents = load_all_urls(urls=filtered_urls)

    texts = process_documents(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    if texts:
        logger.info("Creating embeddings. May take some minutes...")
        store.add_documents(texts)

    logger.info("Ingestion complete")
