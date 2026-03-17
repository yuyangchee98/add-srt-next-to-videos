import argparse
import sys
from pathlib import Path

from .core import (
    transcribe, translate_segments, romanize_segments,
    segments_to_srt, segments_to_dual_srt, segments_to_triple_srt,
    srt_path_for, parse_sub_spec, spec_target_lang, VIDEO_EXTENSIONS,
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

  # Japanese with romaji
  add-srt-next-to-videos ./videos -l ja --subs ja-romaji

  # Japanese + romaji + English translation
  add-srt-next-to-videos ./videos -l ja --subs ja-romaji-en

  # Everything at once
  add-srt-next-to-videos ./videos -l ja --subs ja,en,ja-en,ja-romaji,ja-romaji-en
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
             "Romaji: romaji, ja-romaji, ja-romaji-en (Japanese romanization). "
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

    # Collect all target languages we need to translate to (avoid duplicating work)
    target_langs = set()
    needs_romaji = False
    for spec in sub_specs:
        target = spec_target_lang(spec, source_lang)
        if target:
            target_langs.add(target)
        _, romaji, _ = parse_sub_spec(spec)
        if romaji:
            needs_romaji = True

    needs_llm = len(target_langs) > 0

    print(f"Found {len(videos)} video(s)")
    print(f"Whisper: {WHISPER_MODEL}")
    print(f"Subtitles: {', '.join(sub_specs)}")
    if needs_llm:
        print(f"LLM: {LLM_URL} ({LLM_MODEL})")
    if needs_romaji:
        print(f"Romaji: enabled (pykakasi)")
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
        print(f"  transcribing...", end="\r")
        segments = transcribe(video, language=source_lang)
        non_empty = sum(1 for s in segments if s["text"].strip())
        print(f"  transcribed: {non_empty} segments")

        # Romanize once if any spec needs it
        romaji_segs = None
        if needs_romaji:
            romaji_segs = romanize_segments(segments)
            print(f"  romanized: {len(romaji_segs)} segments")

        # Translate once per target language, cache results
        def progress(msg, end="\n"):
            print(msg, end=end, flush=True)

        translations = {}
        for target in target_langs:
            needed = any(
                spec_target_lang(s, source_lang) == target
                for s in specs_to_do
            )
            if needed:
                translations[target] = translate_segments(
                    segments, source_lang or "auto", target,
                    print_progress=progress,
                )

        # Generate output files
        for spec in specs_to_do:
            srt_path = srt_path_for(video, spec)
            try:
                source, romaji, target = parse_sub_spec(spec)

                if romaji and target:
                    # Triple: ja-romaji-en
                    srt_content = segments_to_triple_srt(segments, romaji_segs, translations[target])
                elif romaji and source:
                    # Dual: ja-romaji
                    srt_content = segments_to_dual_srt(segments, romaji_segs)
                elif romaji:
                    # Single: romaji
                    srt_content = segments_to_srt(romaji_segs)
                elif target:
                    # Dual: ja-en
                    srt_content = segments_to_dual_srt(segments, translations[target])
                elif source == source_lang or spec == "auto":
                    # Single source: ja
                    srt_content = segments_to_srt(segments)
                else:
                    # Single translated: en
                    srt_content = segments_to_srt(translations[source])

                srt_path.write_text(srt_content, encoding="utf-8")
                print(f"  {spec}: {srt_path.name}")
            except Exception as e:
                print(f"  {spec}: error - {e}", file=sys.stderr)

    print("\nDone.")
