# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Viterbox is a Vietnamese Text-to-Speech system with zero-shot voice cloning, based on Chatterbox architecture fine-tuned with 3000+ hours of Vietnamese audio data.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Web UI (Gradio) - opens at http://localhost:7860
python app.py

# CLI inference
python inference.py --text "Xin chào" --output output.wav

# OpenAI-compatible TTS server
python openai_server/tagged_tts_server.py
```

## Architecture

**Three-Stage Pipeline:**
```
Text → T3 (Text-to-Token) → S3Gen (Token-to-Speech) → Waveform (24kHz)
```

1. **T3** (`viterbox/models/t3/`): 520M parameter Transformer converts text to discrete speech tokens with speaker conditioning
2. **S3Gen** (`viterbox/models/s3gen/`): Flow-matching vocoder converts tokens to waveform, includes HiFiGAN decoder
3. **Voice Encoder** (`viterbox/models/voice_encoder/`): Extracts speaker embeddings from reference audio for voice cloning

**Main Entry Points:**
- `viterbox/tts.py` - Core `Viterbox` class with `generate()`, `save_audio()`, `prepare_conditionals()`
- `app.py` - Gradio web interface
- `inference.py` - CLI tool (also registered as `viterbox` command)
- `openai_server/tagged_tts_server.py` - Async server with tag support for silence, soundtracks, speaker switching

## Key Audio Processing Functions (viterbox/tts.py)

- `vad_trim()` - Silero VAD for precise speech endpoint detection
- `apply_fade_out/fade_in()` - Cosine fades to prevent click artifacts
- `apply_dereverberation()` - Spectral gating for room noise reduction
- `crossfade_concat()` - Blends sentence audio segments
- `normalize_text()` - Vietnamese text normalization (requires `soe-vinorm`)

## Generation Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `exaggeration` | 0.0-2.0 | Expression intensity |
| `cfg_weight` | 0.0-1.0 | Voice adherence (classifier-free guidance) |
| `temperature` | 0.1-1.0 | Sampling randomness |
| `sentence_pause_ms` | 0-2000 | Pause between sentences |
| `dereverberation_strength` | 0.0-1.0 | Room noise reduction |

## Tagged TTS Server Tags

The server at `openai_server/tagged_tts_server.py` supports:
- `[silence Ns]` - Insert N seconds of silence
- `[soundtrack Ns]` or `[soundtrack]` - Background music with fade-out
- `[speaker_id]` - Switch voice mid-synthesis (loads from `speakers/` or `wavs/`)
- `[...overlay]` - Overlay soundtrack on speech instead of sequential

## Model Files

Downloaded automatically from HuggingFace (`dolly-vn/viterbox`) to `pretrained/`:
- `t3_ml24ls_v2.safetensors` - T3 model
- `s3gen.pt` - Vocoder
- `ve.pt` - Voice encoder
- `tokenizer_vi_expanded.json` - Text tokenizer

## Commit Convention

Use conventional commits: `feat:`, `fix:`, `chore:`
