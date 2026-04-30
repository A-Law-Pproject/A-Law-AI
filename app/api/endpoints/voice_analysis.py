"""
Compatibility wrapper for the previous mixed voice endpoint module.

Use `app.api.endpoints.voice_standalone` for new standalone HTTP work.
"""
from app.api.endpoints.voice_standalone import (
    analyze_voice,
    analyze_voice_legacy,
    analyze_voice_s3,
    analyze_voice_s3_legacy,
    legacy_router,
    router,
)
