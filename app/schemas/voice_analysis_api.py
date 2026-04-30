"""
Compatibility wrapper for the previous mixed standalone voice schema module.

Use `app.schemas.voice_standalone` for new standalone HTTP work.
"""
from app.schemas.voice_standalone import (
    VoiceAnalysisResponse,
    VoiceAnalysisSummary,
    VoiceAnalyzeS3Request,
    VoiceRiskItem,
)
