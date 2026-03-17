import argparse
import sys
from pathlib import Path

from .core import (
    transcribe, translate_segments, segments_to_srt, segments_to_dual_srt,
    srt_path_for, is_dual, parse_sub_spec, VIDEO_EXTENSIONS,
    WHISPER_MODEL, LLM_URL, LLM_MODEL,
)


def find_video_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return [path]
        return []
    files = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(
        prog="add-srt-next-to-videos",
        description="Generate .srt subtitle files next to video files using local MLX Whisper.",
        epilog="""examples:
  # Transcribe Japanese audio -> Japanese subtitles
  add-srt-next-to-videos ./videos -l ja --subs ja

  # English subtitles only (translated via LLM)
  add-srt-next-to-videos ./videos -l ja --subs en

  # Multiple languages
  add-srt-next-to-videos ./videos -l ja --subs ja,en,zh

  # Dual-language subtitles (source + translation in one file)
  add-srt-next-to-videos ./videos -l ja --subs ja-en

  # Everything at once
  add-srt-next-to-videos ./videos -l ja --subs ja,en,ja-en,ja-zh
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Video file or directory to process recursively",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Source language code (e.g. ja, en, zh). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--subs", "-s",
        default=None,
        help="Comma-separated subtitle specs to generate. "
             "Single language: ja, en, zh. "
             "Dual-language: ja-en, ja-zh (source on top, translation below). "
             "Source language is transcribed via Whisper; others translated via LLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .srt files",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    videos = find_video_files(args.path)
    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    source_lang = args.language

    # Default to source language transcription if no --subs given
    sub_specs = [s.strip() for s in args.subs.split(",")] if args.subs else [source_lang or "auto"]

    # Check if any spec needs LLM (any target != source)
    needs_llm = any(
        (is_dual(s) or s != source_lang) and s != "auto"
        for s in sub_specs
    )

    # Collect all target languages we need to translate to (avoid duplicating work)
    target_langs = set()
    for spec in sub_specs:
        if is_dual(spec):
            _, target = parse_sub_spec(spec)
            target_langs.add(target)
        elif spec != source_lang and spec != "auto":
            target_langs.add(spec)

    print(f"Found {len(videos)} video(s)")
    print(f"Whisper: {WHISPER_MODEL}")
    print(f"Subtitles: {', '.join(sub_specs)}")
    if needs_llm:
        print(f"LLM: {LLM_URL} ({LLM_MODEL})")
    print()

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        # Check if any output is needed for this video
        specs_to_do = []
        for spec in sub_specs:
            srt_path = srt_path_for(video, spec)
            if srt_path.exists() and not args.overwrite:
                print(f"  {spec}: skipped (exists)")
            else:
                specs_to_do.append(spec)

        if not specs_to_do:
            continue

        # Transcribe once
        segments = transcribe(video, language=source_lang)

        # Translate once per target language, cache results
        translations = {}
        for target in target_langs:
            needed = any(
                (is_dual(s) and parse_sub_spec(s)[1] == target) or s == target
                for s in specs_to_do
            )
            if needed:
                translations[target] = translate_segments(
                    segments, source_lang or "auto", target,
                )

        # Generate output files
        for spec in specs_to_do:
            srt_path = srt_path_for(video, spec)
            try:
                if is_dual(spec):
                    _, target = parse_sub_spec(spec)
                    srt_content = segments_to_dual_srt(segments, translations[target])
                elif spec == source_lang or spec == "auto":
                    srt_content = segments_to_srt(segments)
                else:
                    srt_content = segments_to_srt(translations[spec])

                srt_path.write_text(srt_content, encoding="utf-8")
                print(f"  {spec}: {srt_path.name}")
            except Exception as e:
                print(f"  {spec}: error - {e}", file=sys.stderr)

    print("\nDone.")
