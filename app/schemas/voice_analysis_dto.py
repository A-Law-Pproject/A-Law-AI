"""
Compatibility wrapper for the previous mixed voice async DTO module.

Use `app.schemas.voice_contract_fact_check` for new Spring async job work.
"""
from app.schemas.voice_contract_fact_check import (
    FactCheckItem,
    VoiceAnalysisRequest,
    VoiceContractFactCheckRequest,
    VoiceContractFactCheckResult,
    VoiceFactCheckResult,
)
