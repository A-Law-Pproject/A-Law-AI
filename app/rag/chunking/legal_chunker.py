from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LEGAL_SEPARATORS = ["\n\n제", "\n제", "\n\n", "\n", ".", " "]


def create_legal_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> RecursiveCharacterTextSplitter:
    """법률 문서 전용 텍스트 분할기 생성.

    Args:
        chunk_size: 청크 최대 길이.
        chunk_overlap: 청크 간 겹침 길이.

    Returns:
        RecursiveCharacterTextSplitter 인스턴스.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LEGAL_SEPARATORS,
        length_function=len,
    )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """JSONL 문서를 청킹. 이미 조항 단위이므로 초과 크기만 추가 분할.

    Args:
        docs: Document 리스트.
        chunk_size: 청크 최대 길이.
        chunk_overlap: 청크 간 겹침 길이.

    Returns:
        청킹된 Document 리스트.
    """
    splitter = create_legal_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for doc in docs:
        if len(doc.page_content) > chunk_size:
            chunks.extend(splitter.split_documents([doc]))
        else:
            chunks.append(doc)

    return chunks
