# celery_app.py — Celery 앱 초기화
from celery import Celery

celery_app = Celery(
    "contract_analyzer",
    broker="amqp://alawuser:alaw@localhost:5672/",  # RabbitMQ를 브로커로 사용
    backend="db+postgresql://alawuser:alaw@localhost:5432/alawdb"  # 결과 저장소
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    task_track_started=True,             # STARTED 상태 추적
    task_acks_late=True,                 # 작업 완료 후에만 Ack (핵심)
    worker_prefetch_multiplier=1,        # 1개씩만 가져옴
)

# Spring Boot가 publish한 큐를 Celery가 소비
celery_app.conf.task_routes = {
    "tasks.analyze_contract": {"queue": "contract-analysis-queue"}
}