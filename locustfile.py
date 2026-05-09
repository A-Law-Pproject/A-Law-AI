"""
A-LAW FastAPI 부하 테스트
대상: 가천대 임대차 계약 서비스 (DAU 75, MAU 500)

실행:
    locust -f locustfile.py --host http://localhost:8001

Web UI: http://localhost:8089

프로필:
    ALAW_LOCUST_PROFILE=smoke  → 빠른 사전 점검
    ALAW_LOCUST_PROFILE=staged → 기본 단계형 부하 (기본값)
    ALAW_LOCUST_PROFILE=peak   → 공격적 피크 테스트
"""
import os
import random

from locust import HttpUser, LoadTestShape, task, between, events


# ──────────────────────────────────────────
# 테스트 데이터
# ──────────────────────────────────────────

RAG_QUESTIONS = [
    "보증금 반환 기한은 언제인가요?",
    "임대차 계약 갱신은 어떻게 하나요?",
    "전세 계약 시 확정일자를 받아야 하나요?",
    "월세 연체 시 임대인이 계약을 해지할 수 있나요?",
    "임차인의 수선 요청 권리는 어디까지인가요?",
    "계약 만료 후 묵시적 갱신이란 무엇인가요?",
    "보증금을 올려달라는 임대인 요구를 거부할 수 있나요?",
]

RISK_CLAUSES = [
    "보증금 반환은 임대인의 확인 후 30일 이내에 이루어지며, 이 기간 동안의 이자는 지급하지 않는다.",
    "임차인은 퇴거 시 다음 세입자를 직접 구해야 하며, 구하지 못할 경우 보증금 반환이 3개월 유예된다.",
    "계약기간 종료 후 임차인이 원상복구를 하지 않을 경우 보증금에서 전액 공제할 수 있다.",
    "임대인은 언제든지 1개월 전 통보로 계약을 해지할 수 있으며 임차인은 이의를 제기할 수 없다.",
    "임차인은 임대인 동의 없이 어떠한 시설도 변경할 수 없으며 위반 시 보증금을 몰취한다.",
]

CHAT_MESSAGES = [
    "전세 계약 시 확인해야 할 사항은 무엇인가요?",
    "임대차보호법에서 임차인을 보호하는 핵심 조항을 알려주세요.",
    "계약서에 특약사항을 어떻게 작성해야 안전한가요?",
    "전입신고와 확정일자의 차이가 무엇인가요?",
]

LOAD_PROFILES = {
    "smoke": [
        {"cumulative": 30, "users": 2, "spawn_rate": 1},
        {"cumulative": 90, "users": 4, "spawn_rate": 1},
        {"cumulative": 150, "users": 6, "spawn_rate": 1},
    ],
    "staged": [
        {"cumulative": 120, "users": 3, "spawn_rate": 1},
        {"cumulative": 420, "users": 10, "spawn_rate": 1},
        {"cumulative": 780, "users": 20, "spawn_rate": 2},
    ],
    "peak": [
        {"cumulative": 60, "users": 5, "spawn_rate": 1},
        {"cumulative": 240, "users": 15, "spawn_rate": 2},
        {"cumulative": 480, "users": 30, "spawn_rate": 3},
    ],
}

ACTIVE_PROFILE = os.getenv("ALAW_LOCUST_PROFILE", "staged").strip().lower()
ACTIVE_STAGES = LOAD_PROFILES.get(ACTIVE_PROFILE, LOAD_PROFILES["staged"])


# ──────────────────────────────────────────
# 사용자 시나리오
# ──────────────────────────────────────────

class ContractUser(HttpUser):
    """
    임대차 계약 서비스 이용자 시뮬레이션

    task 가중치:
        chat_query(3)    : AI 챗봇 대화 (법률 질의 통합)
        detect_risk(2)   : 계약 조항 독소조항 검사
    """
    wait_time = between(5, 15)  # 실제 사용자 Think Time (초)

    def on_start(self):
        """테스트 시작 전 워밍업: KURE 모델 로드 및 Qdrant 연결 확인"""
        with self.client.get("/ai/contracts/health", catch_response=True, name="[warmup] /ai/contracts/health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"워밍업 실패: {resp.status_code}")

    @task(2)
    def detect_risk(self):
        """독소조항 탐지 - 3회 임베딩 + Qdrant 검색 + LLM"""
        clause = random.choice(RISK_CLAUSES)
        with self.client.post(
            "/ai/contracts/detect-risk",
            json={"clause_text": clause},
            catch_response=True,
            name="/ai/contracts/detect-risk",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                # 응답 유효성 간단 검증
                if "risk_delta" not in body:
                    resp.failure("응답에 risk_delta 누락")
                else:
                    resp.success()
            elif resp.status_code == 503:
                resp.failure("RAG 시스템 초기화 실패 (503)")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(3)
    def chat_query(self):
        """챗봇 단발 질의 - 비동기 병렬 검색 + LLM"""
        message = random.choice(CHAT_MESSAGES)
        with self.client.post(
            "/ai/chat",
            json={"message": message},
            catch_response=True,
            name="/ai/chat",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("RAG 시스템 초기화 실패 (503)")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────
# 단계적 부하 형태 (StepLoadShape)
# ──────────────────────────────────────────

class StepLoadShape(LoadTestShape):
    """환경변수 프로필에 따라 단계형 부하를 적용한다."""

    stages = ACTIVE_STAGES

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["cumulative"]:
                return stage["users"], stage["spawn_rate"]
        return None  # 테스트 종료


# ──────────────────────────────────────────
# 이벤트 훅: 테스트 완료 시 요약 출력
# ──────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    total = stats.total
    print("\n" + "=" * 60)
    print("📊 A-LAW 부하 테스트 결과 요약")
    print(f"프로필: {ACTIVE_PROFILE}")
    print("=" * 60)
    print(f"  총 요청 수     : {total.num_requests:,}")
    print(f"  실패 요청 수   : {total.num_failures:,}")
    print(f"  에러율         : {total.fail_ratio * 100:.1f}%")
    print(f"  RPS (최대)     : {total.max_rps:.2f}")
    print(f"  P50 응답시간   : {total.get_response_time_percentile(0.50):.0f} ms")
    print(f"  P95 응답시간   : {total.get_response_time_percentile(0.95):.0f} ms")
    print(f"  P99 응답시간   : {total.get_response_time_percentile(0.99):.0f} ms")
    print("=" * 60)
    if total.get_response_time_percentile(0.95) > 3000:
        print("⚠️  P95 > 3000ms: KURE 임베딩 병목 발생 구간 Grafana에서 확인")
    if total.fail_ratio > 0.01:
        print("⚠️  에러율 > 1%: OpenAI Rate Limit 또는 서버 과부하 가능성")
    print()
