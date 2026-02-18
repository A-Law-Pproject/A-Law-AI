"""
Celery 설정 및 앱 인스턴스
"""
from celery import Celery
from app.core.config import settings

# Celery 앱 생성
celery_app = Celery(
    "alaw_worker",
    broker=settings.RABBITMQ_URL,
    backend=f"redis://{settings.REDIS_URL.split('://')[-1]}",  # Redis를 결과 백엔드로 사용
    include=["app.services.worker"]  # 태스크 모듈
)

# Celery 설정
celery_app.conf.update(
    # 태스크 설정
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,

    # 재시도 설정
    task_acks_late=True,  # 태스크 완료 후 ack
    task_reject_on_worker_lost=True,

    # 우선순위 큐 설정
    task_queue_max_priority=3,
    task_default_priority=1,

    # 동시성 설정
    worker_prefetch_multiplier=1,  # Worker당 1개씩만 prefetch
    worker_concurrency=3,  # 3개의 워커 스레드

    # 결과 만료 시간
    result_expires=3600,  # 1시간

    # 태스크 라우팅
    task_routes={
        "tasks.analyze_contract": {"queue": settings.ANALYSIS_QUEUE},
        "tasks.analyze_contract_parallel": {"queue": settings.ANALYSIS_QUEUE}
    }
)
