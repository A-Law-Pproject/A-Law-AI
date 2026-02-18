# """
# Worker 실행 엔트리포인트

# 실행 방법:
#     python -m app.services.worker_runner
# """
# import asyncio
# import signal
# import sys

# from loguru import logger

# from app.services.worker import AnalysisWorker


# def setup_logging():
#     """로깅 설정"""
#     logger.remove()
#     logger.add(
#         sys.stdout,
#         format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
#                "<level>{level: <8}</level> | "
#                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
#                "<level>{message}</level>",
#         level="INFO"
#     )
#     logger.add(
#         "logs/worker_{time}.log",
#         rotation="100 MB",
#         retention="7 days",
#         level="DEBUG"
#     )


# async def main():
#     """메인 함수"""
#     setup_logging()

#     worker = AnalysisWorker()

#     # 시그널 핸들러 설정 (graceful shutdown)
#     loop = asyncio.get_event_loop()

#     def shutdown_handler():
#         logger.info("Received shutdown signal")
#         asyncio.create_task(worker.shutdown())

#     # Windows에서는 SIGTERM이 없으므로 예외 처리
#     try:
#         loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
#         loop.add_signal_handler(signal.SIGINT, shutdown_handler)
#     except NotImplementedError:
#         # Windows에서는 signal handler가 지원되지 않음
#         pass

#     logger.info("=" * 50)
#     logger.info("A-LAW Analysis Worker Starting...")
#     logger.info("=" * 50)

#     try:
#         await worker.start()
#     except KeyboardInterrupt:
#         logger.info("Keyboard interrupt received")
#         await worker.shutdown()
#     except Exception as e:
#         logger.error(f"Worker error: {e}")
#         await worker.shutdown()
#         raise


# if __name__ == "__main__":
#     asyncio.run(main())
