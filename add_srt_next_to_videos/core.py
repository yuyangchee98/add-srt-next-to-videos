import os
from pathlib import Path

import mlx_whisper
import requests
from dotenv import load_dotenv

load_dotenv()

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".vob",
}

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")
LLM_URL = os.getenv("LLM_URL", "http://localhost:1234")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-32b-mlx")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_path_for(video_path: Path, sub_spec: str) -> Path:
    """Return path like video.ja.srt, video.en.srt, or video.ja-en.srt."""
    return video_path.with_suffix(f".{sub_spec}.srt")


def is_dual(sub_spec: str) -> bool:
    """Check if a sub spec is dual-language (e.g. 'ja-en')."""
    return "-" in sub_spec


def parse_sub_spec(sub_spec: str) -> tuple[str, str | None]:
    """Parse 'ja' -> ('ja', None), 'ja-en' -> ('ja', 'en')."""
    if "-" in sub_spec:
        parts = sub_spec.split("-", 1)
        return parts[0], parts[1]
    return sub_spec, None


def transcribe(video_path: Path, model: str | None = None, language: str | None = None) -> list[dict]:
    """Transcribe audio with Whisper. Returns segments with timestamps."""
    kwargs = {
        "path_or_hf_repo": model or WHISPER_MODEL,
        "verbose": False,
        "task": "transcribe",
    }
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(video_path), **kwargs)
    return result["segments"]


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using an OpenAI-compatible API (LM Studio, Ollama, etc.)."""
    resp = requests.post(
        f"{LLM_URL}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following {source_lang} text to {target_lang}. "
                               "Return ONLY the translation, nothing else.",
                },
                {"role": "user", "content": text + " /no_think"},
            ],
            "temperature": 0.1,
        },
        timeout=None,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def translate_segments(segments: list[dict], source_lang: str, target_lang: str) -> list[dict]:
    """Translate segment texts via LLM, preserving timestamps."""
    translated = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        t = translate_text(text, source_lang, target_lang)
        translated.append({**seg, "text": t})
    return translated


def segments_to_srt(segments: list[dict]) -> str:
    """Build single-language SRT."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        if not text:
            continue
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_dual_srt(source_segments: list[dict], translated_segments: list[dict]) -> str:
    """Build dual-language SRT: source text on top, translation below."""
    lines = []
    for i, (src, tgt) in enumerate(zip(source_segments, translated_segments), 1):
        src_text = src["text"].strip()
        tgt_text = tgt["text"].strip()
        if not src_text:
            continue
        start = format_timestamp(src["start"])
        end = format_timestamp(src["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(src_text)
        lines.append(tgt_text)
        lines.append("")
    return "\n".join(lines)
