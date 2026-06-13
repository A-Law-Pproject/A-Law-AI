<div align="center">

# A-Law AI 서버

> 임대차 계약서 OCR · 법률 조항 분석 · 독소조항 위험 탐지 AI 백엔드

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-1C3C3C)](https://www.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?logo=pinecone&logoColor=white)](https://www.pinecone.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)](https://openai.com)
[![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-AMQP-FF6600?logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com)

</div>

---

**A-Law AI 서버**는 임대차 계약서 분석 플랫폼 A-Law의 AI 전담 백엔드입니다.  
계약서 OCR 처리, 독소조항 위험 탐지, RAG 기반 법률 챗봇, 음성 팩트체크, 개인정보 마스킹을 담당하며, Spring Boot 메인 서버(포트 8080)와 RabbitMQ를 통해 비동기로 연동됩니다.

---

## 목차

- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택과 선택 이유](#기술-스택과-선택-이유)
- [RAG 파이프라인 상세](#rag-파이프라인-상세)
- [해결한 핵심 문제들](#해결한-핵심-문제들)
- [Spring Boot 연동](#spring-boot-연동)
- [API 엔드포인트](#api-엔드포인트)
- [모니터링 및 평가](#모니터링-및-평가)
- [개발 환경 설정](#개발-환경-설정)
- [배포](#배포)

---

## 주요 기능

### 계약서 OCR 및 구조화
AWS S3에 업로드된 계약서 이미지 또는 직접 업로드된 파일을 Upstage Document Parse API로 분석합니다. 단순 텍스트 추출을 넘어 단어별 좌표 정보를 함께 추출하여, 인식된 텍스트를 원본 계약서 이미지 위에 오버레이로 표시하거나 특정 영역만 선택적으로 마스킹할 수 있습니다.

### 독소조항 위험 탐지
계약서 전체를 조항 단위로 분리하고 각 조항의 위험도를 분석합니다. 정규식 기반 선필터로 명확한 독소조항을 즉시 탐지하고, 판단이 필요한 조항만 GPT-4o와 벡터 검색으로 심층 분석합니다. 최종적으로 조항별 위험 점수, 위험 근거 법령, 개선 권고안을 포함한 리포트를 생성합니다.

### RAG 기반 법률 챗봇
임대차 관련 법률 질문에 검색 증강 생성(RAG)으로 응답합니다. 단순 키워드 검색이 아닌, 질문을 다각도로 확장하고 5개 벡터 컬렉션(법령, 판례, 계약서, 독소조항 사례)을 동시에 검색한 뒤 재정렬과 컨텍스트 압축을 거쳐 근거 있는 답변을 생성합니다.

### 개인정보 마스킹
계약서에 포함된 주민번호, 전화번호, 계좌번호, 상세 주소 등 9개 카테고리의 개인정보를 텍스트와 이미지 두 단계로 마스킹합니다. OCR 결과의 단어 좌표를 활용해 이미지 상의 민감 정보를 검은색 사각형으로 덮어 저장합니다.

### 음성 팩트체크
계약 협의 중 녹음된 음성 파일을 Whisper STT로 전사하고, 실제 계약서 조항과 비교해 발화 내용과 계약 내용의 불일치를 탐지합니다. 원본 음성 파일의 무결성 검증을 위한 해시값도 함께 생성합니다.

### Spring Boot 비동기 연동
Spring Boot로부터 RabbitMQ 메시지를 수신해 OCR 결과 조회 → 요약 생성 → 위험 분석을 비동기로 처리하고, 결과를 다시 RabbitMQ로 발행합니다. 요약과 위험 분석은 `asyncio.gather`로 병렬 실행해 처리 시간을 절반으로 줄였습니다.

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│              사용자 / 프론트엔드                            │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTPS
                           ▼
┌─────────────────────────────────────────────────────────┐
│         Spring Boot 메인 서버 (포트 8080)                  │
│   인증 · 계약서 CRUD · 분석 작업 관리 · 결과 저장            │
└──────────┬───────────────────────────┬──────────────────┘
           │ RabbitMQ (계약서 분석 요청)  │ REST (OCR, 채팅, 위험 탐지)
           ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│           A-Law AI 서버 (포트 8001)                       │
│         FastAPI + LangGraph + LangChain                  │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  OCR     │ │  RAG     │ │ 위험탐지  │ │  음성    │  │
│  │ Upstage  │ │Chatbot   │ │ LangGraph │ │  STT     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              PII 마스킹 레이어                       │  │
│  │  텍스트(정규식) + 이미지(OCR 좌표 기반)               │  │
│  └───────────────────────────────────────────────────┘  │
└──────────┬────────────┬──────────┬────────────┬─────────┘
           │            │          │            │
           ▼            ▼          ▼            ▼
     Pinecone       MongoDB      Redis       AWS S3
    (5개 namespace) (OCR결과)  (세션캐시)  (계약서이미지)
     법령/판례/사례   Atlas      TTL 3600s
```

---

## 기술 스택과 선택 이유

### FastAPI — 비동기 AI 서버의 요구사항

**선택 이유**: AI 처리 파이프라인은 LLM 호출(2~5초), 임베딩 생성(100~200ms), Pinecone 검색(300~500ms) 등 I/O 대기 구간이 길다. Flask나 Django의 동기 모델에서는 하나의 요청이 LLM을 기다리는 동안 다른 요청을 처리하지 못한다. FastAPI의 `async/await` 네이티브 지원으로 동시 사용자 20~30명이 서로를 블로킹하지 않고 처리된다.

또한 Pydantic v2 기반의 스키마 정의가 Spring Boot DTO와의 필드 매핑 검증에 유용했다. `model_config = ConfigDict(populate_by_name=True)`를 활용해 Python의 snake_case 필드명과 Spring Boot의 camelCase 필드명을 동일 스키마에서 처리한다.

### KURE-v1 — 한국어 법률 도메인 임베딩

**선택 이유**: 법률 문서 검색에서 임베딩의 언어 특화는 결과 품질에 직접적인 영향을 미친다. 범용 다국어 모델(multilingual-e5 등)은 "임대차보호법 제6조의3"처럼 한국어 법령 고유 표현을 적절히 임베딩하지 못하는 경우가 잦았다.

KURE-v1(nlpai-lab/KURE-v1)은 1024차원의 한국어 특화 모델로, OpenAI text-embedding-ada-002 대비 한국어 법률 질의 유사도 검색에서 더 높은 정밀도를 보였다. GPU 없이 CPU에서도 동작(배포 서버에 GPU 없음)하고 모델 파일이 약 400MB 수준이어서 Docker 이미지 크기도 관리 가능하다.

단, 단일 스레드에서만 안전하게 동작하므로 `threading.Lock`으로 동시 임베딩 호출을 직렬화하고 Prometheus로 대기 시간을 측정한다.

### Pinecone — 관리형 벡터 DB와 네임스페이스 분리

**선택 이유**: 초기에는 Qdrant를 로컬 컨테이너로 운영했으나 두 가지 문제가 있었다.

첫째, 법령/판례/계약서/독소조항 등 성격이 다른 문서들을 하나의 인덱스에 넣으면 검색 결과가 혼재된다. Pinecone의 네임스페이스 기능으로 5개 컬렉션을 물리적으로 분리하면서도 단일 인덱스로 관리한다.

둘째, Qdrant 자체 운영 시 메모리 관리와 재시작 문제가 발생했다. Pinecone은 서버리스로 인프라를 위임하고 메타데이터 필터(`law_name`, `law_type`, `risk_level`)를 검색 시점에 적용할 수 있어 법령 종류별 필터링이 간결해졌다.

### LangGraph — 도구 호출 루프의 명시적 제어

**선택 이유**: 단순 LangChain 체인으로는 도구 선택과 호출의 흐름을 제어하기 어려웠다. "법령 검색을 몇 번까지 반복할 것인가", "OOS(Out-of-Scope) 질문은 어디서 거절할 것인가" 같은 분기를 코드로 명시하기 어렵고, 디버깅도 힘들다.

LangGraph의 상태 머신 모델은 각 노드(검색, 도구 선택, 응답 생성)를 명시적으로 정의하고 엣지로 흐름을 제어한다. 챗봇 파이프라인은 5개 노드, 위험 분석 파이프라인은 도구 기반 루프(최대 2회)로 구성했다. 특히 무한 도구 호출 루프 방지와 타임아웃 처리가 명확해졌다.

### RabbitMQ — Spring Boot와의 비동기 메시지 연동

**선택 이유**: Spring Boot가 사용자에게 "분석 요청을 접수했습니다"라고 응답한 뒤, AI 처리(OCR + 요약 + 위험 분석)가 백그라운드에서 진행되어야 한다. AI 처리 전체가 평균 10~20초 소요되므로 HTTP 동기 호출은 타임아웃 위험이 있다.

RabbitMQ를 메시지 브로커로 사용하면 Spring Boot는 메시지 발행 후 즉시 응답을 반환하고, AI 서버는 큐에서 순서대로 처리한다. AI 처리 실패 시 최대 3회 재시도하고, JSONDecodeError나 ValidationError처럼 재시도해도 의미 없는 오류는 Dead Letter Queue로 이동한다.

`aio-pika`를 선택한 이유는 FastAPI의 비동기 이벤트 루프와 자연스럽게 통합되기 때문이다. 동기 pika 라이브러리는 별도 스레드에서 실행해야 하는데, 이는 FastAPI의 `async`와 맞지 않는다.

### Redis — 두 가지 캐싱 계층

**선택 이유**: 서로 다른 두 용도로 사용한다.

**세션 캐시**: 챗봇 대화 이력을 TTL 3600초로 저장한다. 인메모리 딕셔너리 대신 Redis를 쓰는 이유는 서버 재시작 시 세션이 유지되고, 향후 수평 확장 시 여러 인스턴스가 동일한 세션에 접근할 수 있기 때문이다.

**쿼리 확장 캐시**: 동일한 질의에 대한 HyDE 문서 생성과 멀티쿼리 확장 결과를 캐싱한다. 쿼리 확장은 LLM 호출을 포함하므로 캐시 히트 시 응답 시간이 약 1초 단축된다.

### MongoDB Atlas — OCR 결과 비정형 저장소

**선택 이유**: OCR 결과는 "단어 텍스트 + 페이지 번호 + 좌표(left, top, width, height, percent 여부)" 형태의 비정형 중첩 구조다. PostgreSQL 같은 관계형 DB에 저장하려면 여러 테이블과 복잡한 조인이 필요하다.

MongoDB는 이 구조를 스키마 변경 없이 그대로 저장한다. 특히 마스킹 메타데이터(어떤 단어의 어느 좌표를 마스킹했는지)도 동일 문서에 함께 저장해 단일 쿼리로 조회 가능하다. Atlas는 TLS 기반 인증서 연결을 지원해 별도 VPN 없이 클라우드에서 안전하게 접속한다.

### BGE CrossEncoder — 멀티컬렉션 점수 정규화

**선택 이유**: 5개 네임스페이스에서 Dense 벡터 검색으로 각각 상위 k개 결과를 가져오면, 네임스페이스마다 점수 분포가 다르다. 법령 컬렉션의 0.78점과 판례 컬렉션의 0.82점은 단순 수치 비교가 불가능하다.

BGE CrossEncoder(BAAI/bge-reranker-v2-m3)는 질의와 각 문서 쌍을 직접 비교해 0~1 범위의 관련성 점수를 계산한다. 이를 통해 서로 다른 컬렉션의 결과를 동일 기준으로 정렬한다. 또한 주거용/상가용 도메인 프리픽스를 쿼리에 주입해 도메인 특화 재정렬을 수행한다.

### Upstage Document Parse — 계약서 레이아웃 분석

**선택 이유**: 임대차 계약서에는 표 형태의 특약 조항, 다단 레이아웃, 날인 영역 등이 포함된다. 범용 OCR(Tesseract 등)은 텍스트를 추출하지만 표의 행/열 구조나 섹션 경계를 인식하지 못한다.

Upstage Document Parse는 레이아웃을 분석해 마크다운 형태로 구조화된 텍스트를 반환한다. 표는 마크다운 테이블로, 제목은 헤딩으로 변환되어 이후 LLM 처리가 용이해진다. 단어별 좌표도 함께 반환해 PII 마스킹의 이미지 처리에 그대로 활용한다.

---

## RAG 파이프라인 상세

```
사용자 질의
    │
    ▼
[1단계] 도메인 분류
    • 질의에서 법령명, 상가 키워드, 세금 키워드 등을 감지
    • 4개 도메인 분류 (주거/상가/세금/일반)
    • LLM 호출 없이 규칙 기반 처리 (0ms)
    │
    ▼
[2단계] 쿼리 확장 (선택)
    • HyDE: "이 질문에 대한 법조문 형식 답변"을 LLM이 생성 → 해당 문서로 검색
    • Multi-Query: 질의를 구어체, 법률용어, 법령명 명시 3가지로 변형
    • Redis 캐시로 동일 질의 반복 LLM 호출 방지
    │
    ▼
[3단계] 병렬 멀티컬렉션 검색
    • Dense: Pinecone 5개 네임스페이스 동시 검색 (asyncio.gather)
    • BM25: 로컬 JSONL 코퍼스 보완 검색 (키워드 정밀 매칭)
    • RRF 융합: Reciprocal Rank Fusion (k=60)으로 두 결과 통합
    • 조문번호 정확 매칭 가중치: 쿼리의 "제X조"가 일치하면 +4~7점
    │
    ▼
[4단계] 재정렬 및 필터링
    • BGE CrossEncoder로 질의-문서 관련성 직접 점수화
    • 도메인 불일치 페널티 적용 (상가 질의 → 주거 문서 -5~20점)
    • 첫 100자 기준 중복 제거
    • 400자 이상 문서는 LLM Contextual Compression으로 압축
    │
    ▼
[5단계] 응답 생성
    • tiktoken으로 토큰 수 계산 후 컨텍스트 구성
    • GPT-4o 호출 (30초 타임아웃)
    • 응답 내 "법령명 제X조" 인용 자동 검증
    •  미검증 인용에 "(미검증)" 주석 또는 번호 제거
    • 소스 문서 마크다운 포맷으로 함께 반환
```

### 독소조항 선필터

LLM 호출 전 정규식으로 명확한 독소조항을 즉시 판별한다. 7개 위험 패턴 중 하나라도 매칭되면 LLM 없이 ClauseRisk를 반환하고, 3개 안전 패턴에 매칭되면 위험 없음으로 즉시 처리한다. LLM이 필요한 조항만 벡터 검색과 GPT-4o 분석으로 처리해 전체 계약서 분석 시간을 단축한다.

| 패턴 | 위험 점수 |
|------|----------|
| 계약갱신청구권 포기 강요 | 88 |
| 보증금 반환 거부/지연 허용 | 90 |
| 임대인 단독 즉시 퇴거 조항 | 85 |
| 우선변제권/임차권등기 포기 | 85 |
| 보증금 전액 몰수 조항 | 87 |
| 전체 수선비 임차인 부담 | 62 |
| 자연마모 원상복구 의무화 | 58 |

---

## 해결한 핵심 문제들

### 문제 1: 멀티컬렉션 검색 결과의 점수 불일치

**상황**: 법령 컬렉션과 판례 컬렉션에서 각각 코사인 유사도로 검색하면 점수 분포가 다르다. 법령 검색 상위 결과가 0.75점이고 판례 검색 상위 결과가 0.82점이라도, 실제 관련성은 법령 쪽이 더 높을 수 있다.

**해결**: 두 단계 접근을 사용했다.
- **RRF(Reciprocal Rank Fusion)**: 절대 점수 대신 순위를 기준으로 통합. 각 컬렉션에서의 순위를 `1/(k+rank)`로 변환해 합산하면 점수 스케일 차이 없이 순위를 통합할 수 있다.
- **BGE CrossEncoder 재정렬**: RRF 통합 후 상위 후보들을 CrossEncoder로 재점수화. 질의와 각 문서를 함께 입력받아 0~1 절대값으로 관련성을 평가하므로 컬렉션 간 비교가 가능해진다.

### 문제 2: LLM 동시 호출로 인한 서버 과부하

**상황**: 계약서 분석 시 조항이 10개면 LLM을 10번 동시 호출한다. OpenAI API Rate Limit에 걸리거나, 단일 서버가 동시에 너무 많은 메모리를 사용해 불안정해졌다.

**해결**: `app/core/ai_runtime.py`에 Semaphore 기반 동시성 제한을 구현했다.
- LLM 최대 동시 호출: 설정값 기준 (기본 3)
- 임베딩 최대 동시 호출: 1 (KURE-v1 스레드 안전 문제)
- 재정렬 최대 동시 호출: 설정값 기준
- 각 대기 시간을 Prometheus 히스토그램으로 측정해 병목을 식별한다

### 문제 3: 법령 조문 번호 환각

**상황**: GPT-4o가 "주택임대차보호법 제6조의4"처럼 실제로 존재하지 않는 조문을 인용하는 환각이 발생했다. 법률 서비스에서 잘못된 법령 인용은 심각한 문제다.

**해결**: `annotate_unverified_citations()` 함수를 구현했다.
1. 응답 텍스트에서 "법령명 제X조" 패턴을 정규식으로 모두 추출
2. 검색에서 가져온 소스 문서들에서 해당 조문 번호가 실제로 등장하는지 확인
3. 소스에 없는 인용은 "(미검증)" 주석을 추가하거나 조문 번호를 제거
4. `rag_legal_citation_total` Prometheus 메트릭으로 검증율을 추적

### 문제 4: 상가/주거 혼합 검색 결과

**상황**: "보증금을 돌려받지 못하면 어떻게 하나요?"라는 질문에 주거용 임대차 답변과 상가 임대차 답변이 뒤섞여 반환됐다. 두 법령의 세부 조항이 다르므로 혼재된 정보는 오답으로 이어진다.

**해결**: 두 레이어로 도메인을 분리했다.
- **검색 단계**: `domain_classifier.py`가 질의를 주거/상가/세금/일반으로 분류하고 Pinecone 메타데이터 필터에 적용한다.
- **재정렬 단계**: BGE CrossEncoder 입력 쿼리 앞에 `"주거용: "` 또는 `"상가용: "` 프리픽스를 붙여 도메인 프리픽스가 있는 문서를 더 높게 점수화한다.
- **페널티 단계**: 도메인 불일치 문서는 최종 점수에서 5~20점을 차감한다.

### 문제 5: OCR 이후 개인정보 유출 위험

**상황**: 계약서 OCR 결과를 MongoDB에 저장할 때 주민번호, 계좌번호, 상세 주소 등 개인정보가 그대로 저장되는 문제가 있었다.

**해결**: 저장 전 두 단계 마스킹을 적용한다.
- **텍스트 마스킹**: 9개 카테고리 정규식으로 텍스트 추출본에서 PII를 감지하고 `●●●●` 형태로 치환한다. 레이블 기반 인식(`주민번호: 900101-1234567` 형태)도 지원해 정확도를 높인다.
- **이미지 마스킹**: OCR에서 추출한 단어별 좌표(백분율)를 활용해 감지된 단어 위치에 검은색 사각형을 그려 masked 이미지를 S3 `masked/` 폴더에 별도 저장한다.
- 마스킹 메타데이터(어느 위치를 마스킹했는지)도 MongoDB에 함께 저장해 감사 추적을 지원한다.

### 문제 6: 음성 파일 원본성 검증

**상황**: 계약 분쟁 시 증거로 제출할 음성 파일이 편집되지 않았음을 증명해야 하는 요구사항이 있었다.

**해결**: `hash_service.py`에서 업로드된 음성 파일의 SHA-256 해시를 계산해 MongoDB에 저장한다. 이후 동일 파일을 다시 제출하면 해시를 비교해 파일 변조 여부를 판별한다.

### 문제 7: 긴 계약서의 컨텍스트 윈도우 초과

**상황**: 임대차 계약서 전체 텍스트가 GPT-4o의 128k 토큰 컨텍스트 내에 들어가더라도, 실제로 분석에 필요한 내용은 특정 조항뿐이다. 불필요한 내용이 많으면 LLM의 주의가 분산되어 답변 품질이 저하된다.

**해결**: `compress_documents()` 함수에서 400자 이상의 긴 검색 결과를 LLM Contextual Compression으로 압축한다. "이 질문과 관련된 부분만 남기고 나머지는 제거하라"는 지시로 각 문서에서 관련 문장만 추출한다. `asyncio.gather`로 복수 문서를 병렬 압축해 추가 지연을 최소화한다.

---

## Spring Boot 연동

A-Law 서비스는 두 서버가 역할을 분담한다.

| 역할 | Spring Boot (8080) | FastAPI (8001) |
|------|-------------------|--------------------|
| 인증/세션 | JWT 발급·검증 | - |
| 계약서 CRUD | PostgreSQL 저장 | - |
| 분석 작업 관리 | 작업 상태 추적 | - |
| OCR 처리 | S3 업로드·키 전달 | Upstage API 호출 |
| AI 분석 | 결과 수신·저장 | RAG, 위험 탐지 |
| 챗봇 | 프록시 전달 | 세션 관리·응답 |

### RabbitMQ 메시지 계약

**Spring → FastAPI (계약서 분석 요청)**
```
Exchange: contract-analysis-ex
Queue:    contract-analysis-queue
Routing:  contract.analyze

{
  "job_id": "uuid",
  "contract_id": 123,
  "s3_key": "contracts/123/document.jpg",
  "user_id": 456
}
```

**FastAPI → Spring (분석 결과)**
```
Exchange: contract.analysis.result
Queue:    contract-analysis-result-queue
Routing:  ai.result

{
  "jobId": "uuid",
  "contractId": 123,
  "status": "COMPLETED",
  "summary": {
    "title": "표준임대차계약서",
    "summaryText": "...",
    "keyTerms": ["보증금", "임대료"],
    "duration": "2024-01-01 ~ 2026-01-01"
  },
  "riskAnalysis": {
    "totalClauses": 12,
    "overallRiskScore": 35,
    "overallRiskLevel": "MEDIUM",
    "clauseResults": [...]
  },
  "processingTimeMs": 18500,
  "completedAt": "2024-12-01T10:30:00"
}
```

> **중요**: Exchange/Queue 이름, DTO 필드명 변경 시 FastAPI `app/core/config.py`와 Spring `RabbitMQConfig.java` 및 관련 DTO를 반드시 동시에 수정해야 한다.

---

## API 엔드포인트

### 챗봇

| Method | Path | 설명 |
|--------|------|------|
| POST | `/ai/chat` | RAG 기반 법률 질의응답 |
| GET | `/ai/chat/{session_id}/history` | 대화 이력 조회 |
| DELETE | `/ai/chat/{session_id}` | 세션 삭제 |

```json
POST /ai/chat
{
  "session_id": "user-123",
  "message": "계약갱신요구권을 거절할 수 있는 경우는?",
  "contract_context": "..."
}
```

### 계약서 분석

| Method | Path | 설명 |
|--------|------|------|
| POST | `/ai/contracts/ocr` | S3 키로 OCR 처리 |
| POST | `/ai/contracts/ocr/full` | 이미지 직접 업로드 OCR |
| POST | `/ai/contracts/explain/term` | 법률 용어 쉬운말 설명 |
| POST | `/ai/contracts/detect-risk` | 독소조항 위험 탐지 |
| GET | `/ai/contracts/health` | RAG 시스템 상태 |

### 음성

| Method | Path | 설명 |
|--------|------|------|
| POST | `/ai/voice/analyze` | 음성 파일 분석 + 팩트체크 |

### 시스템

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 + RabbitMQ 상태 |

---

## 모니터링 및 평가

### Prometheus 메트릭

RAG 파이프라인 각 단계의 지연 시간을 히스토그램으로 측정한다.

| 메트릭 | 설명 |
|--------|------|
| `rag_embedding_latency_seconds` | KURE-v1 임베딩 생성 시간 |
| `rag_llm_latency_seconds` | GPT-4o 응답 시간 |
| `rag_pipeline_stage_latency_seconds` | 단계별 레이턴시 (검색/재정렬/압축) |
| `ai_concurrency_wait_seconds` | Semaphore 대기 시간 |
| `ai_response_cache_total` | 캐시 히트/미스/오류 수 |
| `rag_legal_citation_total` | 법령 인용 포함 응답 수 |
| `rag_refusal_response_total` | 회피성 응답 수 (부정 지표) |
| `rag_reranker_score` | CrossEncoder 점수 분포 |

### RAG 평가 지표

`tests/eval_unified.py`로 하이퍼파라미터 조합별 성능을 측정한다.

| 지표 | 설명 | 목표 |
|------|------|------|
| Hit@3 | 상위 3개 안에 정답 문서 포함 비율 | ≥ 0.70 |
| MRR | 평균 역순위 (정답 문서 순위) | ≥ 0.60 |
| RAGAS Faithfulness | 응답이 컨텍스트에 근거한 비율 | ≥ 0.75 |
| 법령 인용율 | 응답에 법령 인용 포함 비율 | ≥ 0.60 |
| 거부율 | "모르겠습니다" 등 회피 응답 비율 | ≤ 0.10 |

### Grafana 대시보드

`docker-compose.monitoring.yml`으로 Prometheus + Grafana + PushGateway를 로컬에서 실행한다.

```bash
docker-compose -f docker-compose.monitoring.yml up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## 개발 환경 설정
### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 인프라 (RabbitMQ, Redis)
docker-compose up -d rabbitmq redis
```

### RAG 성능 평가

```bash
# 통합 평가 (baseline vs reranker vs query_expansion)
python tests/eval_unified.py --preset all

# RAGAS 평가
python tests/eval_ragas_qna.py

# 부하 테스트 (staged 프로필, 3~20 동시 사용자)
locust -f locustfile.py --host=http://localhost:8001 --users 20 --spawn-rate 3
```

---

## 배포

### Docker

```bash
# 2단계 빌드 (builder: pip + KURE-v1 다운로드, runtime: 실행 환경)
docker build -t alaw-ai-server .

# 실행
docker run -p 8001:8001 --env-file .env alaw-ai-server
```

2단계 빌드를 사용하는 이유: Stage 1에서 PyTorch CPU-only 버전 설치와 KURE-v1 모델 다운로드를 수행하고, Stage 2에서는 pip 패키지와 모델 파일만 복사해 최종 이미지 크기를 줄인다.

### CI/CD (GitHub Actions)

1. **Lint**: flake8으로 문법 오류 검사
2. **Docker Build**: 캐시 활용 이미지 빌드 → DockerHub 푸시
3. **배포 트리거**: A-Law-Cloud 레포에 `deploy-fastapi` repository_dispatch 이벤트 발행

---

## 벡터 DB 컬렉션 구조

| 네임스페이스 | 내용 | 주요 메타데이터 |
|------------|------|----------------|
| `law_database` | 법률 조문, 판례 요약 | `law_name`, `article`, `case_no` |
| `law_statutes` | 주택임대차보호법 등 전문 | `law_name`, `law_type`, `source_dir` |
| `contracts` | 표준 계약서 템플릿 | `title`, `contract_type` |
| `special_clauses_illegal` | 독소조항 실제 사례 | `category`, `risk_level` |
| `special_clauses_normal` | 정상 조항 사례 | `category`, `usage_frequency` |

---

## 성능 특성

| 구간 | 평균 소요 시간 |
|------|--------------|
| KURE-v1 임베딩 (CPU) | 100~200ms |
| Pinecone Dense 검색 | 200~400ms |
| BM25 로컬 검색 | 10~30ms |
| BGE CrossEncoder 재정렬 | 50~100ms |
| GPT-4o LLM 호출 | 2~5초 |
| **전체 RAG 응답** | **3~7초** |
| RabbitMQ 계약서 분석 (병렬) | 10~20초 |
| 동시 사용자 (부하 테스트 기준) | 20~30명 |
