# # ============================================================
# # [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# # services/vector_store.py(구 API) 기반 래퍼입니다.
# # 새로운 구현: app/rag/retriever/multi_retriever.py (멀티 컬렉션 검색)
# # TODO: 안정화 후 삭제 예정
# # ============================================================
# """
# 공유 RAG 서비스 (LEGACY)
# from typing import List, Dict, Optional
# from langchain_core.documents import Document
# from loguru import logger
# from app.services.vector_store import get_vector_store, QdrantVectorStore
# from app.core.config import settings


# class SharedRAGService:
#     """
#     모든 Agent가 공유하는 RAG 서비스

#     주요 기능:
#     - 법률 문서 검색
#     - 컨텍스트 생성
#     - 캐싱 지원
#     """

#     _instance = None

#     def __new__(cls):
#         """싱글톤 패턴 구현"""
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def __init__(self):
#         """초기화"""
#         if self._initialized:
#             return

#         self.vector_store: QdrantVectorStore = get_vector_store()
#         self._cache: Dict[str, List[Document]] = {}
#         self._cache_enabled = True
#         self._max_cache_size = 100

#         logger.info("SharedRAGService initialized (singleton)")
#         self._initialized = True

#     def search(
#         self,
#         query: str,
#         k: int = None,
#         filter_dict: Optional[Dict] = None,
#         use_cache: bool = True
#     ) -> List[Document]:
#         """
#         법률 문서 검색

#         Args:
#             query: 검색 쿼리
#             k: 반환할 문서 수 (기본값: settings.TOP_K_DOCUMENTS)
#             filter_dict: 메타데이터 필터
#             use_cache: 캐시 사용 여부

#         Returns:
#             검색된 Document 리스트
#         """
#         k = k or settings.TOP_K_DOCUMENTS

#         # 캐시 키 생성
#         cache_key = self._generate_cache_key(query, k, filter_dict)

#         # 캐시 확인
#         if use_cache and self._cache_enabled and cache_key in self._cache:
#             logger.debug(f"Cache hit for query: {query[:50]}...")
#             return self._cache[cache_key]

#         # 벡터 검색
#         results = self.vector_store.similarity_search(
#             query=query,
#             k=k,
#             filter_dict=filter_dict
#         )

#         # 캐시 저장
#         if use_cache and self._cache_enabled:
#             self._add_to_cache(cache_key, results)

#         return results

#     def search_by_document_type(
#         self,
#         query: str,
#         document_type: str,
#         k: int = None
#     ) -> List[Document]:
#         """
#         특정 문서 타입으로 필터링하여 검색

#         Args:
#             query: 검색 쿼리
#             document_type: 문서 타입 (예: "주거용부동산임대계약서")
#             k: 반환할 문서 수

#         Returns:
#             검색된 Document 리스트
#         """
#         filter_dict = {"document_type": document_type}
#         return self.search(query, k=k, filter_dict=filter_dict)

#     def search_by_article(
#         self,
#         article: str,
#         k: int = None
#     ) -> List[Document]:
#         """
#         특정 조항으로 검색

#         Args:
#             article: 조항명 (예: "제1조", "제2조")
#             k: 반환할 문서 수

#         Returns:
#             검색된 Document 리스트
#         """
#         filter_dict = {"article": article}
#         return self.search(article, k=k, filter_dict=filter_dict)

#     def build_context(
#         self,
#         documents: List[Document],
#         max_length: int = 2000,
#         include_metadata: bool = True
#     ) -> str:
#         """
#         검색된 문서들로 컨텍스트 생성

#         Args:
#             documents: Document 리스트
#             max_length: 최대 길이 (토큰 최적화)
#             include_metadata: 메타데이터 포함 여부

#         Returns:
#             생성된 컨텍스트 문자열
#         """
#         if not documents:
#             return ""

#         context_parts = []
#         current_length = 0

#         for i, doc in enumerate(documents, 1):
#             # 메타데이터 헤더
#             if include_metadata:
#                 header = f"[문서 {i}]"
#                 if doc.metadata.get("article"):
#                     header += f" {doc.metadata['article']}"
#                 if doc.metadata.get("title"):
#                     header += f" - {doc.metadata['title']}"
#                 header += "\n"
#             else:
#                 header = ""

#             # 내용 추가
#             content = doc.page_content

#             # 길이 체크
#             part_length = len(header) + len(content)
#             if current_length + part_length > max_length:
#                 # 남은 공간만큼만 추가
#                 remaining = max_length - current_length
#                 if remaining > 100:  # 최소 100자는 있어야 의미가 있음
#                     content = content[:remaining - len(header) - 3] + "..."
#                     context_parts.append(header + content)
#                 break

#             context_parts.append(header + content)
#             current_length += part_length

#         return "\n\n".join(context_parts)

#     def get_relevant_context(
#         self,
#         query: str,
#         k: int = None,
#         max_context_length: int = 2000
#     ) -> str:
#         """
#         쿼리에 대한 관련 컨텍스트 바로 생성 (검색 + 컨텍스트 생성 통합)

#         Args:
#             query: 검색 쿼리
#             k: 검색할 문서 수
#             max_context_length: 최대 컨텍스트 길이

#         Returns:
#             생성된 컨텍스트 문자열
#         """
#         documents = self.search(query, k=k)
#         return self.build_context(documents, max_length=max_context_length)

#     def clear_cache(self):
#         """캐시 클리어"""
#         self._cache.clear()
#         logger.info("RAG cache cleared")

#     def disable_cache(self):
#         """캐시 비활성화"""
#         self._cache_enabled = False
#         logger.info("RAG cache disabled")

#     def enable_cache(self):
#         """캐시 활성화"""
#         self._cache_enabled = True
#         logger.info("RAG cache enabled")

#     def get_cache_stats(self) -> Dict:
#         """
#         캐시 통계 정보

#         Returns:
#             캐시 통계 딕셔너리
#         """
#         return {
#             "enabled": self._cache_enabled,
#             "size": len(self._cache),
#             "max_size": self._max_cache_size
#         }

#     def _generate_cache_key(
#         self,
#         query: str,
#         k: int,
#         filter_dict: Optional[Dict]
#     ) -> str:
#         """캐시 키 생성"""
#         filter_str = str(sorted(filter_dict.items())) if filter_dict else ""
#         return f"{query}|{k}|{filter_str}"

#     def _add_to_cache(self, key: str, value: List[Document]):
#         """캐시에 추가 (LRU 방식)"""
#         # 캐시 크기 제한
#         if len(self._cache) >= self._max_cache_size:
#             # 가장 오래된 항목 제거
#             oldest_key = next(iter(self._cache))
#             del self._cache[oldest_key]

#         self._cache[key] = value


# # 싱글톤 인스턴스 반환 함수
# def get_shared_rag() -> SharedRAGService:
#     """
#     공유 RAG 서비스 인스턴스 반환

#     Returns:
#         SharedRAGService 싱글톤 인스턴스
#     """
#     return SharedRAGService()
# """
