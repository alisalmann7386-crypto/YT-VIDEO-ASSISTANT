import os
import requests
from typing import List, Dict, Any, Optional
import whisper
from pydub import AudioSegment


def format_timestamp(seconds: float) -> str:
    """Converts seconds into a human-readable HH:MM:SS or MM:SS format."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_audio_duration(file_path: str) -> float:
    """Helper function to get the exact duration of an audio file in seconds."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        return 0.0


# ── Local OpenAI Whisper Engine ──────────────────────────────────────────────
def transcribe_chunk_whisper(
    chunk_path: str, 
    model: Any, 
    time_offset: float = 0.0,
    task: str = "transcribe"
) -> tuple[str, list[dict]]:
    """Transcribes a single audio chunk using local Whisper."""
    result = model.transcribe(chunk_path, task=task)
    
    chunk_text = result.get("text", "").strip()
    segments = []
    
    for seg in result.get("segments", []):
        start_sec = seg["start"] + time_offset
        end_sec = seg["end"] + time_offset
        
        segments.append({
            "start": format_timestamp(start_sec),
            "end": format_timestamp(end_sec),
            "start_raw": start_sec,
            "end_raw": end_sec,
            "text": seg["text"].strip()
        })
        
    return chunk_text, segments


# ── Sarvam AI Engine (Indic Languages) ─────────────────────────────────────────
def transcribe_chunk_sarvam(
    chunk_path: str,
    api_key: str,
    time_offset: float = 0.0,
    mode: str = "transcribe"
) -> tuple[str, list[dict]]:
    """Transcribes a single audio chunk using Sarvam AI REST API."""

    url = "https://api.sarvam.ai/speech-to-text"

    if not api_key:
        raise ValueError("SARVAM_API_KEY is missing.")

    headers = {
        "api-subscription-key": api_key
    }

    payload = {
        "model": "saaras:v3",
        "mode": mode,
        "language_code": "unknown",
        "with_timestamps": "true"
    }

    import mimetypes

    mime_type = (
        mimetypes.guess_type(chunk_path)[0]
        or "audio/mpeg"
    )

    try:

        with open(chunk_path, "rb") as audio_file:

            files = {
                "file": (
                    os.path.basename(chunk_path),
                    audio_file,
                    mime_type
                )
            }

            response = requests.post(
                url,
                headers=headers,
                data=payload,
                files=files,
                timeout=120
            )

    except requests.RequestException as e:

        raise RuntimeError(
            f"Could not connect to Sarvam API: {e}"
        )

    # ---------------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------------

    if response.status_code != 200:

        if response.status_code == 401:
            raise RuntimeError(
                "Sarvam authentication failed. "
                "Check SARVAM_API_KEY."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "Sarvam returned 403 Forbidden. "
                "Your API request was rejected. "
                "Check that SARVAM_API_KEY is valid and active."
            )

        raise RuntimeError(
            f"Sarvam API error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    # ---------------------------------------------------------
    # PARSE RESPONSE
    # ---------------------------------------------------------

    data = response.json()

    chunk_text = data.get(
        "transcript",
        ""
    ).strip()

    segments = []

    timestamps_obj = data.get("timestamps")

    # ---------------------------------------------------------
    # WORD / SEGMENT TIMESTAMPS
    # ---------------------------------------------------------

    if timestamps_obj and isinstance(
        timestamps_obj,
        dict
    ):

        words_or_chunks = timestamps_obj.get(
            "words",
            []
        )

        for entry in words_or_chunks:

            start_sec = (
                float(
                    entry.get(
                        "start_time_seconds",
                        0.0
                    )
                )
                + time_offset
            )

            end_sec = (
                float(
                    entry.get(
                        "end_time_seconds",
                        0.0
                    )
                )
                + time_offset
            )

            text = (
                entry.get("word", "")
                or entry.get("transcript", "")
            ).strip()

            if not text:
                continue

            segments.append({
                "start": format_timestamp(start_sec),
                "end": format_timestamp(end_sec),
                "start_raw": start_sec,
                "end_raw": end_sec,
                "text": text
            })

    # ---------------------------------------------------------
    # FALLBACK: CHUNK LEVEL TIMESTAMP
    # ---------------------------------------------------------

    if not segments:

        chunk_duration = get_audio_duration(
            chunk_path
        )

        segments.append({
            "start": format_timestamp(
                time_offset
            ),
            "end": format_timestamp(
                time_offset + chunk_duration
            ),
            "start_raw": time_offset,
            "end_raw": (
                time_offset + chunk_duration
            ),
            "text": chunk_text
        })

    return chunk_text, segments


# ── Processing Engine ────────────────────────────────────────────────────────
def process_transcription(
    chunks: List[str],
    provider: str = "whisper",
    model_size: str = "base",
    translate_to_english: bool = False,
    sarvam_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Main orchestrator function to transcribe audio chunks."""
    full_text_list = []
    all_segments = []
    cumulative_offset = 0.0
    
    whisper_model = None
    if provider == "whisper":
        whisper_model = whisper.load_model(model_size)
        
    for chunk_path in chunks:
        if not os.path.exists(chunk_path):
            continue
            
        if provider == "sarvam":
            if not sarvam_api_key:
                sarvam_api_key = os.getenv("SARVAM_API_KEY")
                if not sarvam_api_key:
                    raise ValueError("SARVAM_API_KEY missing. Provide it or set it in your .env file.")
            
            mode = "translate" if translate_to_english else "transcribe"
            chunk_text, segments = transcribe_chunk_sarvam(
                chunk_path=chunk_path,
                api_key=sarvam_api_key,
                time_offset=cumulative_offset,
                mode=mode
            )
        else:
            task = "translate" if translate_to_english else "transcribe"
            chunk_text, segments = transcribe_chunk_whisper(
                chunk_path=chunk_path,
                model=whisper_model,
                time_offset=cumulative_offset,
                task=task
            )
            
        if chunk_text:
            full_text_list.append(chunk_text)
            all_segments.extend(segments)
            
        cumulative_offset += get_audio_duration(chunk_path)
        
    return {
        "full_text": " ".join(full_text_list),
        "segments": all_segments
    }


# ── App Wrapper ──────────────────────────────────────────────────────────────
def transcribe_all(chunks: List[str], language: str = "english") -> Dict[str, Any]:
    """
    Direct function import required by app.py.
    Maps language options to the transcription provider.
    """
    provider = "sarvam" if language.lower() in ["hinglish", "sarvam"] else "whisper"
    translate_flag = True if language.lower() == "hinglish" else False
    
    return process_transcription(
        chunks=chunks,
        provider=provider,
        translate_to_english=translate_flag
    )