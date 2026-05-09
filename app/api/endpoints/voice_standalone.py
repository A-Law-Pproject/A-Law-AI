"""
Standalone HTTP endpoints for voice analysis that do not depend on Spring async jobs.
"""
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.schemas.voice_standalone import VoiceAnalyzeS3Request, VoiceAnalysisResponse
from app.services.voice.hash_service import download_from_s3, process_and_store_audio, save_voice_analysis_result
from app.services.voice.standalone_analysis_service import summarize_voice_analysis
from app.services.voice.stt_service import transcribe_and_extract

router = APIRouter(prefix="/standalone")
legacy_router = APIRouter()


async def _analyze_uploaded_voice(
    audio_file: UploadFile,
    source_id: str,
) -> VoiceAnalysisResponse:
    start_ms = int(time.time() * 1000)

    try:
        file_bytes = await audio_file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="업로드한 음성 파일이 비어 있습니다.")

        filename = audio_file.filename or "audio.mp3"
        content_type = audio_file.content_type or "audio/mpeg"
        logger.info(
            f"[VoiceStandalone] Upload analyze request: filename={filename}, size={len(file_bytes)}"
        )

        audio_meta = await process_and_store_audio(
            file_bytes=file_bytes,
            original_filename=filename,
            source_id=source_id,
            content_type=content_type,
        )
        segments, agreements = await transcribe_and_extract(
            file_bytes=file_bytes,
            filename=filename,
        )
        transcript = " ".join(seg.text for seg in segments).strip()
        summary = await summarize_voice_analysis(transcript, agreements, segments)

        elapsed_ms = int(time.time() * 1000) - start_ms
        await save_voice_analysis_result({
            "source_id": source_id,
            "audio_hash": audio_meta.file_hash,
            "s3_key": audio_meta.s3_key,
            "transcript": transcript,
            "summary": summary.model_dump(mode="json"),
            "agreements": [a.model_dump(mode="json") for a in agreements],
            "processing_time_ms": elapsed_ms,
        })

        return VoiceAnalysisResponse(
            success=True,
            transcript=transcript,
            summary=summary,
            segments=segments,
            agreements=agreements,
            audio_meta=audio_meta,
            processing_time_ms=elapsed_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VoiceStandalone] Upload analyze failed: {e}")
        return VoiceAnalysisResponse(
            success=False,
            error_message=str(e),
            processing_time_ms=int(time.time() * 1000) - start_ms,
        )


async def _analyze_s3_voice(request: VoiceAnalyzeS3Request) -> VoiceAnalysisResponse:
    start_ms = int(time.time() * 1000)

    try:
        logger.info(f"[VoiceStandalone] S3 analyze request: s3_key={request.s3_key}")
        file_bytes = await download_from_s3(request.s3_key)
        if not file_bytes:
            raise HTTPException(status_code=404, detail="S3 음성 파일을 찾을 수 없습니다.")

        filename = request.s3_key.split("/")[-1]
        segments, agreements = await transcribe_and_extract(
            file_bytes=file_bytes,
            filename=filename,
        )
        transcript = " ".join(seg.text for seg in segments).strip()
        summary = await summarize_voice_analysis(transcript, agreements, segments)

        audio_meta = await process_and_store_audio(
            file_bytes=file_bytes,
            original_filename=filename,
            source_id=request.source_id or "standalone",
            content_type=_ext_to_content_type(filename),
        )

        elapsed_ms = int(time.time() * 1000) - start_ms
        await save_voice_analysis_result({
            "source_id": request.source_id or "standalone",
            "audio_hash": audio_meta.file_hash,
            "s3_key": audio_meta.s3_key,
            "transcript": transcript,
            "summary": summary.model_dump(mode="json"),
            "agreements": [a.model_dump(mode="json") for a in agreements],
            "processing_time_ms": elapsed_ms,
        })

        return VoiceAnalysisResponse(
            success=True,
            transcript=transcript,
            summary=summary,
            segments=segments,
            agreements=agreements,
            audio_meta=audio_meta,
            processing_time_ms=elapsed_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VoiceStandalone] S3 analyze failed: {e}")
        return VoiceAnalysisResponse(
            success=False,
            error_message=str(e),
            processing_time_ms=int(time.time() * 1000) - start_ms,
        )


@router.post(
    "/analyze",
    response_model=VoiceAnalysisResponse,
    summary="Standalone 음성 분석",
)
async def analyze_voice(
    audio_file: UploadFile = File(..., description="분석할 음성 파일"),
    source_id: str = Form(default="standalone", description="메타데이터 분류용 ID"),
) -> VoiceAnalysisResponse:
    return await _analyze_uploaded_voice(audio_file, source_id)


@legacy_router.post(
    "/analyze",
    response_model=VoiceAnalysisResponse,
    include_in_schema=False,
    deprecated=True,
)
async def analyze_voice_legacy(
    audio_file: UploadFile = File(..., description="분석할 음성 파일"),
    source_id: str = Form(default="standalone", description="메타데이터 분류용 ID"),
) -> VoiceAnalysisResponse:
    return await _analyze_uploaded_voice(audio_file, source_id)


@router.post(
    "/analyze-s3",
    response_model=VoiceAnalysisResponse,
    summary="Standalone S3 음성 분석",
)
async def analyze_voice_s3(request: VoiceAnalyzeS3Request) -> VoiceAnalysisResponse:
    return await _analyze_s3_voice(request)


@legacy_router.post(
    "/analyze-s3",
    response_model=VoiceAnalysisResponse,
    include_in_schema=False,
    deprecated=True,
)
async def analyze_voice_s3_legacy(
    request: VoiceAnalyzeS3Request,
) -> VoiceAnalysisResponse:
    return await _analyze_s3_voice(request)


def _ext_to_content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp3"
    mapping = {
        "mp3": "audio/mpeg",
        "mp4": "audio/mp4",
        "m4a": "audio/mp4",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "webm": "audio/webm",
    }
    return mapping.get(ext, "audio/mpeg")
