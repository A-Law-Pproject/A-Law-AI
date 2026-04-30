"""
Compatibility wrapper for the previous mixed async voice consumer module.

Use `app.services.voice_contract_fact_check_consumer` for new Spring async job work.
"""
from app.services.voice_contract_fact_check_consumer import (
    VoiceContractFactCheckConsumer,
    VoiceRabbitMQConsumer,
    start_voice_consumer,
    start_voice_contract_fact_check_consumer,
    stop_voice_consumer,
    stop_voice_contract_fact_check_consumer,
    voice_consumer,
    voice_contract_fact_check_consumer,
)
