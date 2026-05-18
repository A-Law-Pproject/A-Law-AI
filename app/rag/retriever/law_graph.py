"""법령 간 관계 그래프 모듈.

한국 임대차 도메인에서 법령들이 어떻게 연결되는지를 정적 맵으로 표현한다.
런타임에 infer_law_statutes_filter()가 이 맵을 참조해 관련 법령을 OR 필터로 확장한다.

법령 관계 유형:
- supplements: 특별법에 규정 없는 부분을 보충하는 일반법 (민법 등)
- procedural: 집행·소송 절차를 규율하는 법령 (민사집행법 등)
- enforcement: 모법의 위임을 받아 세부사항을 정한 하위 법령 (시행령/시행규칙)
- special_law: 민법에 대한 특별법 관계
"""

from __future__ import annotations

# 법령 계층 및 관계 맵
# key: 법령명, value: {관계유형: [관련 법령명, ...]}
_LAW_GRAPH: dict[str, dict[str, list[str]]] = {
    "주택임대차보호법": {
        "supplements": ["민법"],
        "procedural": ["민사집행법"],
        "enforcement": ["주택임대차보호법 시행령"],
        "related": [
            "전세사기피해자 지원 및 주거안정에 관한 특별법",
            "임차권등기명령 절차에 관한 규칙",
            "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙",
        ],
    },
    "상가건물 임대차보호법": {
        "supplements": ["민법"],
        "procedural": ["민사집행법"],
        "enforcement": ["상가건물 임대차보호법 시행령"],
    },
    "민법": {
        "special_law": ["주택임대차보호법", "상가건물 임대차보호법"],
    },
    "민사집행법": {
        "related": ["주택임대차보호법", "상가건물 임대차보호법", "민법"],
    },
    "전세사기피해자 지원 및 주거안정에 관한 특별법": {
        "supplements": ["주택임대차보호법", "민법"],
    },
    "집합건물의 소유 및 관리에 관한 법률": {
        "supplements": ["민법"],
        "related": ["공동주택관리법"],
    },
}

# 개념별 검색 우선 법령 목록.
# infer_law_statutes_filter()에서 법령명 직접 매칭이 없을 때 개념 키워드로 폴백.
# 각 entry의 첫 번째 법령이 primary, 이후가 관련 법령 (최대 2개까지만 확장).
_CONCEPT_LAW_MAP: dict[str, list[str]] = {
    # 집행·소송 절차
    "보증금_반환_소송": ["주택임대차보호법", "민사집행법", "민법"],
    "경매_배당": ["민사집행법", "주택임대차보호법"],
    "강제집행": ["민사집행법", "민법"],
    "지급명령": ["민사집행법"],
    "내용증명": ["민법", "주택임대차보호법"],
    # 임대차 핵심 권리
    "대항력": ["주택임대차보호법", "민법"],
    "우선변제권": ["주택임대차보호법", "민사집행법"],
    "임차권등기": ["주택임대차보호법", "임차권등기명령 절차에 관한 규칙"],
    "계약갱신요구권": ["주택임대차보호법"],
    "묵시적갱신": ["주택임대차보호법", "민법"],
    # 보증금·차임
    "보증금반환": ["주택임대차보호법", "민법", "민사집행법"],
    "차임증액": ["주택임대차보호법", "민법"],
    # 수선·원상복구
    "수선의무": ["민법", "주택임대차보호법"],
    "원상복구": ["민법", "주택임대차보호법"],
    # 전세사기
    "전세사기": ["전세사기피해자 지원 및 주거안정에 관한 특별법", "주택임대차보호법"],
    # 상가 관련
    "권리금": ["상가건물 임대차보호법"],
    "상가차임증액": ["상가건물 임대차보호법", "민법"],
    "상가보증금반환": ["상가건물 임대차보호법", "민법", "민사집행법"],
}

# 개념 키워드 → 개념 키 매핑 (질의에서 키워드 탐지 후 _CONCEPT_LAW_MAP 룩업)
_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "보증금_반환_소송": ["보증금 반환 소송", "보증금반환소송", "반환청구소송"],
    "경매_배당": ["경매", "배당", "경락"],
    "강제집행": ["강제집행", "집행권원"],
    "지급명령": ["지급명령"],
    "내용증명": ["내용증명"],
    "대항력": ["대항력"],
    "우선변제권": ["우선변제권", "최우선변제"],
    "임차권등기": ["임차권등기명령", "임차권등기"],
    "계약갱신요구권": ["계약갱신요구권", "갱신요구", "갱신청구"],
    "묵시적갱신": ["묵시적 갱신", "묵시갱신"],
    "보증금반환": ["보증금 반환", "보증금반환", "보증금을 돌려"],
    "차임증액": ["차임 증액", "월세 인상", "보증금 증액", "임대료 인상"],
    "수선의무": ["수선의무", "수선 의무", "수리 의무", "보일러", "노후"],
    "원상복구": ["원상복구", "원상회복"],
    "전세사기": ["전세사기", "깡통전세"],
    "권리금": ["권리금"],
    "상가차임증액": ["상가 차임", "상가 임대료"],
    "상가보증금반환": ["상가 보증금"],
}


def detect_concepts(query: str) -> list[str]:
    """질의에서 법령 개념 키워드를 탐지하여 개념 키 목록 반환."""
    normalized = query.replace(" ", "")
    found: list[str] = []
    for concept_key, keywords in _CONCEPT_KEYWORDS.items():
        for kw in keywords:
            if kw.replace(" ", "") in normalized:
                found.append(concept_key)
                break
    return found


def get_related_laws(primary_law: str, query: str = "", max_extra: int = 2) -> list[str]:
    """primary_law와 관련된 추가 법령 목록 반환 (최대 max_extra개).

    Args:
        primary_law: 주 법령명 (infer_law_statutes_filter에서 식별된 법령).
        query: 질의 텍스트 (개념 기반 확장 시 사용).
        max_extra: 추가로 반환할 법령 수 상한. 기본 2.

    Returns:
        primary_law를 제외한 관련 법령 이름 목록 (중복 없음, 최대 max_extra개).
    """
    related: list[str] = []
    seen: set[str] = {primary_law}

    node = _LAW_GRAPH.get(primary_law, {})

    # supplements 관계: 보충 일반법 우선 추가
    for law in node.get("supplements", []):
        if law not in seen:
            related.append(law)
            seen.add(law)

    # procedural 관계: 절차법 추가
    for law in node.get("procedural", []):
        if law not in seen:
            related.append(law)
            seen.add(law)

    # 개념 기반 확장: 질의에서 개념 탐지 → _CONCEPT_LAW_MAP 참조
    if query:
        for concept in detect_concepts(query):
            for law in _CONCEPT_LAW_MAP.get(concept, []):
                if law not in seen:
                    related.append(law)
                    seen.add(law)

    return related[:max_extra]


def build_multi_law_filter(primary_law: str, related_laws: list[str]) -> dict:
    """primary_law + related_laws를 OR로 묶은 Pinecone 메타데이터 필터 생성.

    Args:
        primary_law: 주 법령명.
        related_laws: 추가 법령 이름 목록.

    Returns:
        단일 법령이면 {"law_name": primary_law},
        복수 법령이면 {"$or": [{"law_name": ...}, ...]} 형식.
    """
    if not related_laws:
        return {"law_name": primary_law}

    all_laws = [primary_law] + related_laws
    return {"$or": [{"law_name": law} for law in all_laws]}
