"""
법률 .doc 파일 샘플 업로드 스크립트 (조항 단위 청킹, 5개만)

실행:
    python scripts/sample_upload_law_docs.py

업로드 namespace: law_statutes (law_database와 별도 컬렉션)
"""
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.vector_store.pinecone_adapter import PineconeAdapter
from app.rag.data_loader.doc_loader import load_doc_dir

NAMESPACE = "law_statutes"
SAMPLE_SIZE = 5
DOC_DIR = Path(__file__).parent.parent / "data" / "raw" / "학습법률문서" / "법률"


def main():
    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 1. .doc 파일 로드 및 조항 단위 청킹
    logger.info(f"[1] .doc 파일 로드: {DOC_DIR}")
    all_docs = load_doc_dir(DOC_DIR)
    logger.info(f"    총 {len(all_docs)}개 청크 생성")

    # 2. 샘플 5개 선택
    sample_docs = all_docs[:SAMPLE_SIZE]
    logger.info(f"[2] 샘플 {SAMPLE_SIZE}개 선택")
    for i, doc in enumerate(sample_docs, 1):
        title = doc.metadata.get("title", "")
        article = doc.metadata.get("article", "")
        preview = doc.page_content[:60].replace("\n", " ")
        logger.info(f"    [{i}] {title} / {article} | {preview}...")

    # 3. 임베딩
    logger.info(f"[3] 임베딩 생성 중...")
    embeddings = KUREEmbeddings()
    vectors_list = embeddings.embed_documents([doc.page_content for doc in sample_docs])

    # 4. Pinecone upsert
    logger.info(f"[4] Pinecone namespace='{NAMESPACE}'에 업로드 중...")
    adapter = PineconeAdapter(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX,
    )

    pinecone_vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": vec,
            "metadata": {"content": doc.page_content, **doc.metadata},
        }
        for doc, vec in zip(sample_docs, vectors_list)
    ]

    adapter.upsert(namespace=NAMESPACE, vectors=pinecone_vectors)
    logger.info(f"[완료] {len(pinecone_vectors)}개 벡터 → namespace='{NAMESPACE}'")


if __name__ == "__main__":
    main()
