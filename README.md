# add-srt-next-to-videos

Generate `.srt` subtitle files next to video files. Point it at a file or folder, get subtitles.

Runs entirely on your Mac — [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) handles speech-to-text, a local LLM handles translation. Nothing leaves your machine.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4) — MLX only runs on Apple Silicon
- Python 3.10+
- [pipx](https://pipx.pypa.io/) for system-wide install
- For translation: any local LLM server with an OpenAI-compatible `/v1/chat/completions` endpoint — [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), etc.

## Install

```bash
# Homebrew
brew install yuyangchee98/tap/add-srt-next-to-videos

# From PyPI
pip install add-srt-next-to-videos

# Or with pipx (isolated install, recommended)
pipx install add-srt-next-to-videos

# Or directly from GitHub
pipx install git+https://github.com/yuyangchee98/add-srt-next-to-videos.git
```

After install, create a `.env` file in the directory where you run the command (or your home directory):

```bash
cp .env.example .env
# edit .env with your model preferences
```

To uninstall: `pipx uninstall add-srt-next-to-videos`

For development:
```bash
git clone https://github.com/yuyangchee98/add-srt-next-to-videos.git
cd add-srt-next-to-videos
uv venv && uv pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and edit:

```env
# Whisper model for speech-to-text (HuggingFace repo)
WHISPER_MODEL=mlx-community/whisper-large-v3-mlx

# LLM for translation (any server with an OpenAI-compatible /v1/chat/completions endpoint)
# Works with: LM Studio, Ollama, vLLM, llama.cpp server, etc.
LLM_URL=http://localhost:1234
LLM_MODEL=qwen3-32b-mlx
```

Whisper model weights (~3GB) download automatically on first run and cache in `~/.cache/huggingface/`.

The LLM is only used for translation. If you only need transcription (source language subs), you don't need an LLM server running.

## Usage

```bash
# Transcribe Japanese audio -> Japanese subtitles
add-srt-next-to-videos ./videos -l ja --subs ja

# Translate to English
add-srt-next-to-videos ./videos -l ja --subs en

# Multiple languages at once
add-srt-next-to-videos ./videos -l ja --subs ja,en,zh

# Dual-language subs (both languages in one file, for language learning)
add-srt-next-to-videos ./videos -l ja --subs ja-en

# Mix and match
add-srt-next-to-videos ./videos -l ja --subs ja,en,ja-en,ja-zh

# Single file
add-srt-next-to-videos ./episode01.mp4 -l ja --subs ja,en
```

Works on single files or entire directories (recursive). Existing `.srt` files are skipped unless you pass `--overwrite`.

## Output files

The sub spec determines the filename:

```
video.mp4  ->  video.ja.srt       (transcription)
video.mp4  ->  video.en.srt       (translation)
video.mp4  ->  video.ja-en.srt    (dual: japanese on top, english below)
```

A dual-language `.ja-en.srt` looks like this:

```
1
00:00:05,000 --> 00:00:10,679
母ちゃん母ちゃん今日はどこ行くの
Mom, mom, where are we going today?

2
00:00:10,679 --> 00:00:12,060
美容院よ
The beauty salon.
```

## How it works

1. **Whisper transcribes** the audio once per video — timestamped segments in the source language
2. **LLM translates** each segment's text to other languages (only if needed)

Whisper handles transcription only. All translation goes through your local LLM for better accuracy.

The tool transcribes once and caches the result per video, so `--subs ja,en,ja-en` runs Whisper once and translates to English once, even though two outputs use the English translation.

## CLI reference

```
add-srt-next-to-videos [-h] [--language LANGUAGE] [--subs SUBS] [--overwrite] path
```

| Flag | Short | Default | What it does |
|---|---|---|---|
| `path` | | (required) | Video file or directory (recursive) |
| `--language` | `-l` | auto-detect | Source language (`ja`, `en`, `zh`, etc.) |
| `--subs` | `-s` | source lang | What to generate: `ja`, `en`, `ja-en`, or comma-separated |
| `--overwrite` | | off | Regenerate existing `.srt` files |

Whisper model and LLM settings are configured in `.env` (see [Configuration](#configuration)).

## Supported video formats

`.mp4` `.mkv` `.avi` `.mov` `.webm` `.flv` `.wmv` `.m4v` `.mpg` `.mpeg` `.ts` `.vob`

## Notes

- If a run gets interrupted, just re-run the same command — it skips videos that already have their `.srt` files
- Language codes are ISO 639-1. Whisper supports these languages for transcription:

  `af` Afrikaans, `ar` Arabic, `hy` Armenian, `az` Azerbaijani, `be` Belarusian, `bs` Bosnian, `bg` Bulgarian, `ca` Catalan, `zh` Chinese, `hr` Croatian, `cs` Czech, `da` Danish, `nl` Dutch, `en` English, `et` Estonian, `fi` Finnish, `fr` French, `gl` Galician, `de` German, `el` Greek, `he` Hebrew, `hi` Hindi, `hu` Hungarian, `is` Icelandic, `id` Indonesian, `it` Italian, `ja` Japanese, `kn` Kannada, `kk` Kazakh, `ko` Korean, `lv` Latvian, `lt` Lithuanian, `mk` Macedonian, `ms` Malay, `mr` Marathi, `mi` Maori, `ne` Nepali, `no` Norwegian, `fa` Persian, `pl` Polish, `pt` Portuguese, `ro` Romanian, `ru` Russian, `sr` Serbian, `sk` Slovak, `sl` Slovenian, `es` Spanish, `sw` Swahili, `sv` Swedish, `tl` Tagalog, `ta` Tamil, `th` Thai, `tr` Turkish, `uk` Ukrainian, `ur` Urdu, `vi` Vietnamese, `cy` Welsh

  Translation to any language depends on your LLM's capabilities

## License

MIT
