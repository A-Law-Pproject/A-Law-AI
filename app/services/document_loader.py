# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# 새로운 구현: app/rag/data_loader/jsonl_loader.py, special_clause_loader.py
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
법률 문서 로더 (LEGACY)
import json
import os
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.documents import Document
from loguru import logger
from app.core.config import settings


class LegalDocumentLoader:
    """법률 문서 로더 클래스"""

    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: 법률 문서 루트 경로
        """
        self.base_path = Path(base_path or settings.LEGAL_DOCS_PATH)
        logger.info(f"Document loader initialized with path: {self.base_path}")

    def load_all_documents(self) -> List[Document]:
        """
        모든 법률 문서 로드

        Returns:
            Document 리스트
        """
        all_documents = []

        # 모든 JSONL 파일 찾기
        jsonl_files = list(self.base_path.rglob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files")

        for jsonl_file in jsonl_files:
            try:
                docs = self.load_jsonl(jsonl_file)
                all_documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {jsonl_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def load_jsonl(self, file_path: Path) -> List[Document]:
        """
        JSONL 파일에서 문서 로드

        Args:
            file_path: JSONL 파일 경로

        Returns:
            Document 리스트
        """
        documents = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        doc = self._parse_document(data, file_path)
                        if doc:
                            documents.append(doc)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at {file_path}:{line_num} - {e}")
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

        return documents

    def _parse_document(self, data: Dict, source_file: Path) -> Optional[Document]:
        """
        JSON 데이터를 Document 객체로 변환

        Args:
            data: JSON 데이터
            source_file: 소스 파일 경로

        Returns:
            Document 객체 또는 None
        """
        try:
            # 필수 필드 확인
            content = data.get("content", "").strip()
            if not content:
                return None

            # 메타데이터 구성
            metadata = {
                "doc_id": data.get("doc_id", ""),
                "document_type": data.get("document_type", ""),
                "chunk_id": data.get("chunk_id", ""),
                "title": data.get("title", ""),
                "article": data.get("article", ""),
                "source_file": str(source_file),
                "content_labels": data.get("content_labels", []),
            }

            # 선택적 필드 추가
            optional_fields = [
                "clause_field",
                "ftc_conclusions",
                "dv_advantageous",
                "unfavorable_provision",
                "case_no",
                "court_name",
                "judgment_date"
            ]

            for field in optional_fields:
                if field in data and data[field]:
                    metadata[field] = data[field]

            # 관련 법령 텍스트 추가
            if "related_law_text" in data and data["related_law_text"]:
                metadata["related_law_text"] = data["related_law_text"]

            # 분석 근거 추가
            if "analysis_basis" in data and data["analysis_basis"]:
                metadata["analysis_basis"] = data["analysis_basis"]

            # Document 객체 생성
            document = Document(
                page_content=content,
                metadata=metadata
            )

            return document

        except Exception as e:
            logger.warning(f"Failed to parse document: {e}")
            return None

    def load_by_type(self, document_type: str) -> List[Document]:
        """
        특정 문서 타입만 로드

        Args:
            document_type: 문서 타입 (예: "주거용부동산임대계약서")

        Returns:
            Document 리스트
        """
        all_docs = self.load_all_documents()
        filtered_docs = [
            doc for doc in all_docs
            if doc.metadata.get("document_type") == document_type
        ]

        logger.info(f"Filtered {len(filtered_docs)} documents of type '{document_type}'")
        return filtered_docs

    def load_by_directory(self, directory_name: str) -> List[Document]:
        """
        특정 디렉토리의 문서만 로드

        Args:
            directory_name: 디렉토리 이름 (예: "계약_Training_라벨링_임대")

        Returns:
            Document 리스트
        """
        target_path = self.base_path / directory_name
        if not target_path.exists():
            logger.warning(f"Directory not found: {target_path}")
            return []

        documents = []
        jsonl_files = list(target_path.rglob("*.jsonl"))

        for jsonl_file in jsonl_files:
            try:
                docs = self.load_jsonl(jsonl_file)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"Loaded {len(documents)} documents from '{directory_name}'")
        return documents

    def get_document_statistics(self) -> Dict:
        """
        문서 통계 정보 반환

        Returns:
            통계 정보 딕셔너리
        """
        all_docs = self.load_all_documents()

        # 문서 타입별 카운트
        type_counts = {}
        for doc in all_docs:
            doc_type = doc.metadata.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        # 평균 문서 길이
        avg_length = sum(len(doc.page_content) for doc in all_docs) / len(all_docs) if all_docs else 0

        statistics = {
            "total_documents": len(all_docs),
            "document_types": type_counts,
            "average_content_length": int(avg_length),
            "unique_types": len(type_counts)
        }

        return statistics


def load_legal_documents() -> List[Document]:
    """
    모든 법률 문서 로드 (편의 함수)

    Returns:
        Document 리스트
    """
    loader = LegalDocumentLoader()
    return loader.load_all_documents()
"""
