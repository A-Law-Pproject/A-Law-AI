"""
법률 문서 전처리 Agent
- 표 처리
- 줄바꿈 문단 처리
- 특수문자 정제
"""
import re
from typing import List
from langchain.schema import Document
from loguru import logger

##  법률 문서 전처리 전담 Agent

class LegalDocumentPreprocessor:

    def __init__(self):
        # 법률 문서 특수 패턴
        self.article_pattern = re.compile(r'제\s*(\d+)\s*조')
        self.section_pattern = re.compile(r'제\s*(\d+)\s*항')
        self.subsection_pattern = re.compile(r'제\s*(\d+)\s*호')

    def process(self, raw_text: str, source: str = "unknown") -> Document:
 
        logger.info(f"Preprocessing document from: {source}")

        # 1. 표 처리
        text = self._process_tables(raw_text)

        # 2. 줄바꿈 및 문단 정리
        text = self._process_paragraphs(text)

        # 3. 특수문자 정제
        text = self._clean_special_characters(text)

        # 4. 연속 공백 제거
        text = self._remove_redundant_spaces(text)

        # 5. 메타데이터 추출
        metadata = self._extract_metadata(text, source)

        return Document(
            page_content=text,
            metadata=metadata
        )

    def _process_tables(self, text: str) -> str:
        """
        표 형식 데이터를 텍스트로 변환

        예시:
        | 항목 | 값 |
        |------|-----|
        | 보증금 | 1억 |

        → "항목: 보증금, 값: 1억"
        """
        # Markdown 스타일 표 감지
        table_pattern = re.compile(
            r'\|(.+)\|[\r\n]+\|[-\s|]+\|[\r\n]+((?:\|.+\|[\r\n]+)+)',
            re.MULTILINE
        )

        def replace_table(match):
            header = match.group(1)
            rows = match.group(2)

            # 헤더 파싱
            headers = [h.strip() for h in header.split('|') if h.strip()]

            # 행 파싱
            processed_rows = []
            for row in rows.strip().split('\n'):
                cells = [c.strip() for c in row.split('|') if c.strip()]
                if len(cells) == len(headers):
                    row_text = ', '.join([f"{h}: {c}" for h, c in zip(headers, cells)])
                    processed_rows.append(row_text)

            return '\n'.join(processed_rows)

        text = table_pattern.sub(replace_table, text)

        # HTML 스타일 표 제거 (복잡도 높음)
        text = re.sub(r'<table>.*?</table>', '[표 생략]', text, flags=re.DOTALL)

        return text

    def _process_paragraphs(self, text: str) -> str:
        """
        줄바꿈 및 문단 정리

        - 불필요한 줄바꿈 제거
        - 조항 구분 유지
        - 문단 간 공백 정규화
        """
        # 조항 앞에 줄바꿈 보장
        text = re.sub(r'([^\n])(제\s*\d+\s*조)', r'\1\n\n\2', text)

        # 연속된 줄바꿈을 2개로 통일 (문단 구분)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 단일 줄바꿈은 공백으로 (문장 연결)
        lines = text.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue

            # 조항 시작은 새로운 문단
            if self.article_pattern.match(line) or \
               self.section_pattern.match(line):
                if processed_lines and processed_lines[-1]:
                    processed_lines.append('')
                processed_lines.append(line)
            else:
                # 일반 텍스트는 이전 줄과 연결
                if processed_lines and processed_lines[-1] and \
                   not processed_lines[-1].endswith(('.', '다', '음')):
                    processed_lines[-1] += ' ' + line
                else:
                    processed_lines.append(line)

        return '\n'.join(processed_lines)

    def _clean_special_characters(self, text: str) -> str:
        """
        특수문자 정제

        - 불필요한 기호 제거
        - 법률 기호 보존 (①, ②, ㉠, ㉡ 등)
        """
        # 보존할 특수문자 패턴
        preserved = r'[①-⑳㉠-㉿]'

        # 제어 문자 제거 (탭, 캐리지 리턴 등)
        text = re.sub(r'[\t\r\f\v]', ' ', text)

        # 불필요한 기호 제거 (단, 법률 기호는 보존)
        text = re.sub(r'[^\w\s.,;:()\[\]{}\"\'<>%/\-+=' + preserved + r'가-힣]', '', text)

        return text

    def _remove_redundant_spaces(self, text: str) -> str:
        """연속 공백 제거"""
        # 연속된 공백을 하나로
        text = re.sub(r' {2,}', ' ', text)

        # 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _extract_metadata(self, text: str, source: str) -> dict:
        """
        메타데이터 추출

        - 조항 번호
        - 문서 카테고리
        - 키워드
        """
        metadata = {
            "source": source,
            "source_type": "legal",
        }

        # 조항 번호 추출 (첫 번째 조항)
        article_match = self.article_pattern.search(text)
        if article_match:
            metadata["article"] = f"제{article_match.group(1)}조"

        # 키워드 추출 (간단한 빈도 기반)
        keywords = self._extract_keywords(text)
        metadata["keywords"] = keywords

        return metadata

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        키워드 추출 (빈도 기반)

        Args:
            text: 텍스트
            top_k: 상위 K개 키워드

        Returns:
            키워드 리스트
        """
        # 법률 용어 사전
        legal_terms = [
            '임대인', '임차인', '보증금', '차임', '계약',
            '갱신', '증액', '해지', '명도', '우선변제권',
            '대항력', '확정일자', '월세', '전세'
        ]

        # 출현 빈도 계산
        term_counts = {}
        for term in legal_terms:
            count = text.count(term)
            if count > 0:
                term_counts[term] = count

        # 상위 K개 추출
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:top_k]]

    def process_batch(self, raw_documents: List[dict]) -> List[Document]:
        """
        배치 처리

        Args:
            raw_documents: [{"text": "...", "source": "..."}, ...]

        Returns:
            전처리된 Document 리스트
        """
        processed_docs = []
        for doc in raw_documents:
            processed = self.process(
                raw_text=doc.get("text", ""),
                source=doc.get("source", "unknown")
            )
            processed_docs.append(processed)

        logger.info(f"Processed {len(processed_docs)} documents in batch")
        return processed_docs
