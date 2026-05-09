"""Upload AI Hub unfavorable lease clause JSON files to Pinecone.

Default namespace:
    lease_clauses_unfavorable

Run:
    python scripts/upload_unfavorable_lease_clauses.py --dry-run
    python scripts/upload_unfavorable_lease_clauses.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.core.config import settings
from app.rag.data_loader.unfavorable_clause_json_loader import (
    load_unfavorable_clause_json_dir,
)


BATCH_SIZE = 100
DEFAULT_NAMESPACE = "lease_clauses_unfavorable"
DEFAULT_SOURCE_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "raw"
    / "학습법률문서"
    / "리스크분석용_법률"
    / "법률_규정_(판결서_약관_등)_텍스트_분석_데이터"
    / "법률_Training_약관"
    / "TL_2"
    / "TL_2.약관"
    / "TL_2.약관"
    / "1.Training"
    / "라벨링데이터"
    / "TL_2.약관"
)


def _vector_id(namespace: str, source_path: str) -> str:
    digest = hashlib.sha1(f"{namespace}:{source_path}".encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def upload_documents(
    adapter: PineconeAdapter,
    embeddings: KUREEmbeddings,
    namespace: str,
    documents: list,
) -> int:
    total = 0
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        vectors = embeddings.embed_documents([doc.page_content for doc in batch])
        pinecone_vectors = []

        for doc, vector in zip(batch, vectors):
            source_path = doc.metadata.get("source_path", doc.metadata.get("source_file", ""))
            pinecone_vectors.append(
                {
                    "id": _vector_id(namespace, source_path),
                    "values": vector,
                    "metadata": {"content": doc.page_content, **doc.metadata},
                }
            )

        adapter.upsert(namespace=namespace, vectors=pinecone_vectors)
        total += len(batch)
        logger.info(f"[{namespace}] uploaded {total}/{len(documents)}")

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()

    logger.info(f"source_dir={source_dir}")
    logger.info(f"namespace={args.namespace}")

    documents = load_unfavorable_clause_json_dir(source_dir)
    if args.limit is not None:
        documents = documents[: args.limit]

    logger.info(f"documents={len(documents)}")
    if documents:
        sample = documents[0]
        logger.info(f"sample title={sample.metadata.get('title')}")
        logger.info(f"sample source_path={sample.metadata.get('source_path')}")
        logger.info(f"sample preview={sample.page_content[:300].replace(chr(10), ' ')}")

    if args.dry_run:
        logger.info("dry-run enabled; no Pinecone upload performed")
        return

    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from app.rag.embedding.kure import KUREEmbeddings
    from app.rag.vector_store.pinecone_adapter import PineconeAdapter

    embeddings = KUREEmbeddings()
    adapter = PineconeAdapter(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX,
    )
    uploaded = upload_documents(adapter, embeddings, args.namespace, documents)
    logger.info(
        f"[done] uploaded {uploaded} vectors to index='{settings.PINECONE_INDEX}', namespace='{args.namespace}'"
    )


if __name__ == "__main__":
    main()
