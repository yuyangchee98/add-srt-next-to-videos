import os
from pathlib import Path

import mlx_whisper
import pykakasi
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

_kakasi = pykakasi.kakasi()


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_path_for(video_path: Path, sub_spec: str) -> Path:
    """Return path like video.ja.srt, video.en.srt, or video.ja-romaji-en.srt."""
    return video_path.with_suffix(f".{sub_spec}.srt")


def parse_sub_spec(sub_spec: str) -> tuple[str | None, bool, str | None]:
    """Parse spec into (source_lang, include_romaji, target_lang).

    'ja'           -> ('ja', False, None)
    'romaji'       -> (None, True, None)
    'ja-en'        -> ('ja', False, 'en')
    'ja-romaji'    -> ('ja', True, None)
    'ja-romaji-en' -> ('ja', True, 'en')
    """
    parts = sub_spec.split("-")
    if len(parts) == 1:
        if parts[0] == "romaji":
            return None, True, None
        return parts[0], False, None
    elif len(parts) == 2:
        if parts[1] == "romaji":
            return parts[0], True, None
        return parts[0], False, parts[1]
    elif len(parts) == 3 and parts[1] == "romaji":
        return parts[0], True, parts[2]
    # fallback: treat first and last as source/target
    return parts[0], False, parts[-1]


def spec_needs_translation(sub_spec: str, source_lang: str | None) -> bool:
    """Check if a spec requires LLM translation."""
    _, _, target = parse_sub_spec(sub_spec)
    if target:
        return True
    source, romaji, _ = parse_sub_spec(sub_spec)
    if not romaji and source != source_lang and sub_spec != "auto" and sub_spec != "romaji":
        return True
    return False


def spec_target_lang(sub_spec: str, source_lang: str | None) -> str | None:
    """Return the LLM translation target language for a spec, or None."""
    source, _, target = parse_sub_spec(sub_spec)
    if target:
        return target
    # Single-language spec that differs from source needs translation
    if not _ and source != source_lang and sub_spec != "auto" and sub_spec != "romaji":
        return source
    return None


def romanize_text(text: str) -> str:
    """Convert Japanese text to romaji using pykakasi."""
    result = _kakasi.convert(text)
    return " ".join(item["hepburn"] for item in result)


def romanize_segments(segments: list[dict]) -> list[dict]:
    """Convert segment texts to romaji, preserving timestamps."""
    romanized = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            romanized.append({**seg, "text": romanize_text(text)})
    return romanized


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


def translate_segments(segments: list[dict], source_lang: str, target_lang: str,
                       print_progress=None) -> list[dict]:
    """Translate segment texts via LLM, preserving timestamps."""
    non_empty = [seg for seg in segments if seg["text"].strip()]
    total = len(non_empty)
    translated = []
    for idx, seg in enumerate(non_empty, 1):
        text = seg["text"].strip()
        if print_progress:
            print_progress(f"  translating to {target_lang}: {idx}/{total}", end="\r")
        t = translate_text(text, source_lang, target_lang)
        translated.append({**seg, "text": t})
    if print_progress:
        print_progress(f"  translating to {target_lang}: {total}/{total} done")
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


def segments_to_dual_srt(source_segments: list[dict], second_segments: list[dict]) -> str:
    """Build dual-language SRT: source text on top, second below."""
    lines = []
    for i, (src, snd) in enumerate(zip(source_segments, second_segments), 1):
        src_text = src["text"].strip()
        snd_text = snd["text"].strip()
        if not src_text:
            continue
        start = format_timestamp(src["start"])
        end = format_timestamp(src["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(src_text)
        lines.append(snd_text)
        lines.append("")
    return "\n".join(lines)


def segments_to_triple_srt(source_segments: list[dict], romaji_segments: list[dict],
                           translated_segments: list[dict]) -> str:
    """Build triple-language SRT: source, romaji, translation."""
    lines = []
    for i, (src, rom, tgt) in enumerate(
        zip(source_segments, romaji_segments, translated_segments), 1
    ):
        src_text = src["text"].strip()
        rom_text = rom["text"].strip()
        tgt_text = tgt["text"].strip()
        if not src_text:
            continue
        start = format_timestamp(src["start"])
        end = format_timestamp(src["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(src_text)
        lines.append(rom_text)
        lines.append(tgt_text)
        lines.append("")
    return "\n".join(lines)
