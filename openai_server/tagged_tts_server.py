#!/usr/bin/env python3
"""
Tagged TTS Server for Viterbox

Processes tagged text directly with support for:
- [silence Ns] - Insert N seconds of silence
- [soundtrack Ns] or [soundtrack] - Insert background music with fade-out
- [speaker_id] - Switch voice for subsequent text

Uses the same OpenAI-compatible /v1/audio/speech endpoint format.
"""

import os
import sys
import time
import tempfile
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import asyncio
import re
from aiohttp import web

import torch
import torchaudio
import numpy as np

from underthesea import sent_tokenize
from langdetect import detect

import soundfile as sf
import librosa

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(APP_DIR)

import logging

def setup_logger(name):
    """Setup a simple logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)

from viterbox import Viterbox

import warnings
warnings.filterwarnings("ignore")

temp_dir = f"{APP_DIR}/cache/temp/"
os.makedirs(temp_dir, exist_ok=True)

# Language mapping
language_dict = {
    'Tiếng Việt': 'vi',
    'Tiếng Việt (Vietnamese)': 'vi',
    'English': 'en',
    'English (US)': 'en',
    'Auto': 'auto',
}

default_language = 'vi'
language_codes = list(language_dict.values())

def lang_detect(text):
    """Detect language from text."""
    try:
        lang = detect(text)
        if lang in language_codes:
            return lang
        if lang.startswith('zh'):
            return 'vi'
        return 'vi'
    except:
        return 'vi'

# Tag regex pattern - same as tagged_tts.py
ALL_TAGS = re.compile(
    r'\[silence\s+(\d+)s?\]|'           # [silence Ns]
    r'\[soundtrack(?:\s+(\d+)s?)?\]|'   # [soundtrack] or [soundtrack Ns]
    r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]',    # [speaker_id]
    re.IGNORECASE
)

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SOUNDTRACK_DURATION = 10.0
DEFAULT_FADE_OUT_DURATION = 5.0
TTS_PADDING_SECONDS = 0.5
TTS_FADE_MS = 20


@dataclass
class TextChunk:
    """A chunk of text to be synthesized with a specific voice."""
    voice: str
    text: str


@dataclass
class SilenceChunk:
    """A chunk of silence."""
    duration: float  # seconds


@dataclass
class SoundtrackChunk:
    """A chunk of soundtrack/background music."""
    duration: float  # seconds
    fade_out: float = DEFAULT_FADE_OUT_DURATION


Chunk = Union[TextChunk, SilenceChunk, SoundtrackChunk]

# Viterbox model
viterbox_model = None

# Speaker conditioning cache: speaker_id -> TTSConds
speaker_conditionals_cache: dict = {}

def load_model():
    global viterbox_model
    viterbox_model = Viterbox.from_pretrained(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Viterbox model loaded successfully")

load_model()

# Directories
speakers_dir = f"{APP_DIR}/speakers"
soundtracks_dir = f"{APP_DIR}/soundtracks"
default_speaker_id = "storyteller_1"

# Thread pool for audio generation
executor = ThreadPoolExecutor(max_workers=4)


def parse_tags(text: str, default_voice: str) -> List[Chunk]:
    """Parse text with tags into a list of chunks.

    Args:
        text: Input text with tags
        default_voice: Default voice ID for text without voice tags

    Returns:
        List of Chunk objects (TextChunk, SilenceChunk, or SoundtrackChunk)
    """
    chunks: List[Chunk] = []
    current_voice = default_voice
    last_end = 0

    QUOTE_CHARS = ['"', "'", '"', '"', "'", "'", '"']

    for match in ALL_TAGS.finditer(text):
        text_before = text[last_end:match.start()].strip()
        if text_before:
            for q in QUOTE_CHARS:
                text_before = text_before.strip(q)
            text_before = text_before.strip()
            if text_before:
                chunks.append(TextChunk(voice=current_voice, text=text_before))

        silence_seconds = match.group(1)
        soundtrack_seconds = match.group(2)
        voice_id = match.group(3)

        if silence_seconds is not None:
            chunks.append(SilenceChunk(duration=float(silence_seconds)))
        elif match.group(0).lower().startswith('[soundtrack'):
            duration = float(soundtrack_seconds) if soundtrack_seconds else DEFAULT_SOUNDTRACK_DURATION
            chunks.append(SoundtrackChunk(duration=duration))
        elif voice_id is not None:
            current_voice = voice_id

        last_end = match.end()

    text_after = text[last_end:].strip()
    if text_after:
        for q in QUOTE_CHARS:
            text_after = text_after.strip(q)
        text_after = text_after.strip()
        if text_after:
            chunks.append(TextChunk(voice=current_voice, text=text_after))

    return chunks


def get_speaker_audio_path(speaker_id: str) -> str:
    """Get reference audio path for a speaker from speakers/ folder."""
    speaker_wav = Path(f"{speakers_dir}/{speaker_id}.wav")
    if speaker_wav.exists():
        return str(speaker_wav)

    speaker_dir = Path(f"{speakers_dir}/{speaker_id}")
    if speaker_dir.exists():
        wav_files = list(speaker_dir.glob("*.wav"))
        if wav_files:
            return str(wav_files[0])

    raise ValueError(f"No reference audio found for speaker: {speaker_id}")


def list_available_speakers() -> list:
    """List all available speakers from the speakers/ folder."""
    speakers = []

    for f in Path(speakers_dir).glob("*.wav"):
        speaker_id = f.stem
        speakers.append({
            'id': speaker_id,
            'source': 'local',
            'cached': True,
            'path': str(f)
        })

    for d in Path(speakers_dir).iterdir():
        if d.is_dir():
            wav_files = list(d.glob("*.wav"))
            if wav_files:
                speakers.append({
                    'id': d.name,
                    'source': 'local',
                    'cached': True,
                    'path': str(wav_files[0])
                })

    return speakers


def get_soundtrack_files() -> List[Path]:
    """Get list of available soundtrack files."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    return [
        f for f in Path(soundtracks_dir).iterdir()
        if f.suffix.lower() in audio_extensions
    ]


def get_speaker_conditionals(speaker_id: str, exaggeration: float = 0.5):
    """Get cached speaker conditionals or extract and cache them.

    Args:
        speaker_id: Speaker identifier
        exaggeration: Expression intensity (0.0 - 2.0)

    Returns:
        TTSConds object with speaker embeddings
    """
    global viterbox_model, speaker_conditionals_cache

    # Check cache first
    if speaker_id in speaker_conditionals_cache:
        logger.debug(f"Using cached conditionals for speaker: {speaker_id}")
        return speaker_conditionals_cache[speaker_id]

    # Extract and cache
    ref_path = get_speaker_audio_path(speaker_id)
    logger.info(f"Extracting conditionals for speaker: {speaker_id}")
    conditionals = viterbox_model.prepare_conditionals(ref_path, exaggeration=exaggeration)
    speaker_conditionals_cache[speaker_id] = conditionals

    logger.info(f"Cached conditionals for speaker: {speaker_id} (cache size: {len(speaker_conditionals_cache)})")
    return conditionals


def clear_speaker_cache(speaker_id: str = None):
    """Clear speaker cache.

    Args:
        speaker_id: Specific speaker to clear, or None to clear all
    """
    global speaker_conditionals_cache
    if speaker_id is not None:
        speaker_conditionals_cache.pop(speaker_id, None)
        logger.info(f"Cleared cache for speaker: {speaker_id}")
    else:
        speaker_conditionals_cache.clear()
        logger.info("Cleared all speaker caches")


def apply_fade(audio: np.ndarray, fade_in_ms: int, fade_out_ms: int, sample_rate: int) -> np.ndarray:
    """Apply fade-in and fade-out to audio to prevent clicks.

    Args:
        audio: Input audio array
        fade_in_ms: Fade-in duration in milliseconds
        fade_out_ms: Fade-out duration in milliseconds
        sample_rate: Sample rate

    Returns:
        Audio with fade applied
    """
    fade_in_samples = int(fade_in_ms / 1000 * sample_rate)
    fade_out_samples = int(fade_out_ms / 1000 * sample_rate)

    if len(audio) <= fade_in_samples + fade_out_samples:
        return audio

    # Fade in
    if fade_in_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        audio[:fade_in_samples] *= fade_in

    # Fade out
    if fade_out_samples > 0:
        fade_out = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        audio[-fade_out_samples:] *= fade_out

    return audio


def generate_silence(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Generate silence of specified duration.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        Silence array
    """
    n_samples = int(duration * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def synthesize_text_chunk(
    text: str,
    speaker_id: str,
    language: str,
    temperature: float = 0.8,
    exaggeration: float = 0.5
) -> np.ndarray:
    """Generate TTS audio for a text chunk with fade and padding.

    Args:
        text: Text to synthesize
        speaker_id: Voice ID
        language: Language code
        temperature: Sampling temperature
        exaggeration: Expression intensity (0.0 - 2.0)

    Returns:
        Audio array with padding and fade applied
    """
    global viterbox_model

    # Skip empty or very short text
    alpha_count = sum(1 for c in text if c.isalpha()) if text else 0
    if alpha_count < 2:
        return generate_silence(TTS_PADDING_SECONDS * 2)

    # Detect language
    lang_code = lang_detect(text) if language == 'auto' else language_dict.get(language, 'vi')

    # Get cached conditionals (this extracts and caches if not already cached)
    conditionals = get_speaker_conditionals(speaker_id, exaggeration=exaggeration)

    # Generate with Viterbox using cached conditionals
    wav_tensor = viterbox_model.generate(
        text=text,
        language=lang_code,
        temperature=temperature,
        cfg_weight=0.5,
        repetition_penalty=2.0,
        split_sentences=True,
        sentence_pause_ms=500,
        dereverberation=True,
        dereverberation_strength=0.5,
    )

    # Convert to numpy
    wav_array = wav_tensor[0].cpu().numpy().astype(np.float32)

    # Apply fade to prevent clicks
    wav_array = apply_fade(wav_array, TTS_FADE_MS, TTS_FADE_MS, DEFAULT_SAMPLE_RATE)

    # Add padding at start and end
    padding_samples = int(TTS_PADDING_SECONDS * DEFAULT_SAMPLE_RATE)
    padding = np.zeros(padding_samples, dtype=np.float32)
    wav_array = np.concatenate([padding, wav_array, padding])

    return wav_array


def load_soundtrack(
    duration: float,
    fade_out: float = DEFAULT_FADE_OUT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Load random soundtrack with fade-out.

    Args:
        duration: Target duration in seconds
        fade_out: Fade-out duration at end (seconds)
        sample_rate: Sample rate

    Returns:
        Audio array with fade-out applied
    """
    soundtrack_files = get_soundtrack_files()

    if not soundtrack_files:
        logger.warning("No soundtrack files found, using silence")
        return generate_silence(duration, sample_rate)

    # Pick a random soundtrack
    soundtrack_file = random.choice(soundtrack_files)
    logger.info(f"Using soundtrack: {soundtrack_file.name}")

    try:
        # Load the soundtrack
        audio_data, sr = sf.read(soundtrack_file)

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)

        audio_data = audio_data.astype(np.float32)

        # Calculate target length
        target_samples = int(duration * sample_rate)

        # Loop or trim to match duration
        if len(audio_data) < target_samples:
            repeats = (target_samples // len(audio_data)) + 1
            audio_data = np.tile(audio_data, repeats)

        # Trim to exact duration
        audio_data = audio_data[:target_samples]

        # Apply fade-out
        fade_samples = int(fade_out * sample_rate)
        if fade_samples > 0 and fade_samples < len(audio_data):
            fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            audio_data[-fade_samples:] *= fade_curve

        return audio_data

    except Exception as e:
        logger.error(f"Error loading soundtrack {soundtrack_file}: {e}")
        return generate_silence(duration, sample_rate)


def overlay_audio(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Overlay overlay audio onto base audio.

    Args:
        base: Base audio array
        overlay: Audio to overlay

    Returns:
        Mixed audio
    """
    if len(overlay) > len(base):
        # Extend base if overlay is longer
        base = np.pad(base, (0, len(overlay) - len(base)), mode='constant')
    elif len(base) > len(overlay):
        # Extend overlay if base is longer
        overlay = np.pad(overlay, (0, len(base) - len(overlay)), mode='constant')

    # Simple mix (average to prevent clipping)
    mixed = (base + overlay) / 2.0
    return mixed


def process_tagged_text(
    tagged_text: str,
    language: str,
    default_voice: str,
    temperature: float = 0.8
) -> Tuple[np.ndarray, int]:
    """Process tagged text and generate audio.

    Args:
        tagged_text: Input text with tags
        language: Language code
        default_voice: Default voice ID
        temperature: Sampling temperature

    Returns:
        Tuple of (audio array, sample rate)
    """
    # Parse tags into chunks
    chunks = parse_tags(tagged_text, default_voice)

    if not chunks:
        return np.array([], dtype=np.float32), DEFAULT_SAMPLE_RATE

    logger.info(f"Processing {len(chunks)} chunks")

    audio_segments: List[np.ndarray] = []
    current_voice = default_voice

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {type(chunk).__name__}")

        if isinstance(chunk, TextChunk):
            current_voice = chunk.voice
            logger.info(f"  Voice: {chunk.voice}")
            logger.info(f"  Text: {chunk.text[:50]}..." if len(chunk.text) > 50 else f"  Text: {chunk.text}")

            audio = synthesize_text_chunk(chunk.text, chunk.voice, language, temperature)
            audio_segments.append(('tts', audio))

        elif isinstance(chunk, SilenceChunk):
            logger.info(f"  Silence: {chunk.duration}s")
            audio = generate_silence(chunk.duration)
            audio_segments.append(('silence', audio))

        elif isinstance(chunk, SoundtrackChunk):
            logger.info(f"  Soundtrack: {chunk.duration}s (fade: {chunk.fade_out}s)")
            audio = load_soundtrack(chunk.duration, chunk.fade_out)
            audio_segments.append(('soundtrack', audio))

    # Combine segments - soundtrack overlays TTS, others concatenate
    final_audio = None

    for seg_type, audio in audio_segments:
        if final_audio is None:
            final_audio = audio
        elif seg_type == 'soundtrack':
            # Overlay soundtrack on existing audio
            final_audio = overlay_audio(final_audio, audio)
        else:
            # Concatenate silence/TTS
            final_audio = np.concatenate([final_audio, audio])

    if final_audio is None:
        return np.array([], dtype=np.float32), DEFAULT_SAMPLE_RATE

    return final_audio, DEFAULT_SAMPLE_RATE


def synthesize_speech(input_text, speaker_id, temperature=0.8, language='vi'):
    """Process text and generate audio using Viterbox."""
    global viterbox_model

    start = time.time()
    logger.info(f"Start processing text: {input_text[:30]}... [length: {len(input_text)}]")
    logger.info(f"Speaker ID: {speaker_id}")

    # Get reference audio path
    ref_path = get_speaker_audio_path(speaker_id)

    # Detect language if auto
    lang_code = lang_detect(input_text) if language == 'auto' else language_dict.get(language, 'vi')

    # Generate with Viterbox
    wav_tensor = viterbox_model.generate(
        text=input_text,
        language=lang_code,
        audio_prompt=ref_path,
        temperature=temperature,
        cfg_weight=0.5,
        repetition_penalty=2.0,
        split_sentences=True,
        sentence_pause_ms=500,
        dereverberation=True,
        dereverberation_strength=0.5,
    )

    end = time.time()
    processing_time = end - start

    # Convert tensor to numpy
    wav_array = wav_tensor[0].cpu().numpy()

    # Calculate approximate tokens
    num_of_tokens = int(len(wav_array) * 50 / 24000)
    tokens_per_second = num_of_tokens / processing_time if processing_time > 0 else 0

    logger.info(f"End processing text: {input_text[:30]}")
    message = f"  {tokens_per_second:.1f} tok/s - {num_of_tokens} tokens - in {processing_time:.2f}s"
    logger.info(message)

    return (viterbox_model.sr, wav_array)


async def handle_speech_request(request):
    """Handles the /v1/audio/speech endpoint with tag support."""
    global default_speaker_id

    try:
        request_data = await request.json()
        text_to_speak = request_data.get('text')
        language = request_data.get('language', 'Tiếng Việt')
        speaker_id = request_data.get('speaker', default_speaker_id)
        temperature = request_data.get('temperature', 0.8)

        if not text_to_speak:
            return web.json_response({"error": "Missing or 'text' field"}, status=400)

        # Check if text contains tags
        has_tags = ALL_TAGS.search(text_to_speak) is not None

        if has_tags:
            logger.info(f"Detected tags in input, using tagged processing")
            logger.info(f"Input: {text_to_speak[:100]}..." if len(text_to_speak) > 100 else f"Input: {text_to_speak}")

            # Validate at least one speaker exists
            try:
                get_speaker_audio_path(speaker_id)
            except ValueError:
                try:
                    get_speaker_audio_path(default_speaker_id)
                    speaker_id = default_speaker_id
                except ValueError:
                    return web.json_response({"error": "No reference audio found for any speaker"}, status=400)

            # Map language to code
            lang_code = language_dict.get(language, 'vi')

            # Process tagged text
            loop = asyncio.get_event_loop()
            wav_array, sample_rate = await loop.run_in_executor(
                executor,
                process_tagged_text,
                text_to_speak,
                lang_code,
                speaker_id,
                temperature
            )

            if len(wav_array) == 0:
                return web.json_response({"error": "Failed to generate audio"}, status=500)

            # Update default speaker from last TextChunk
            for chunk in parse_tags(text_to_speak, speaker_id):
                if isinstance(chunk, TextChunk):
                    default_speaker_id = chunk.voice

        else:
            # Regular processing (no tags)
            try:
                ref_path = get_speaker_audio_path(speaker_id)
            except ValueError:
                try:
                    ref_path = get_speaker_audio_path(default_speaker_id)
                    speaker_id = default_speaker_id
                except ValueError:
                    return web.json_response({"error": "No reference audio found for any speaker"}, status=400)

            logger.info(f"Using speaker: {speaker_id}")

            # Map language to code
            lang_code = language_dict.get(language, 'vi')

            # Generate speech
            sample_rate, wav_array = await asyncio.get_event_loop().run_in_executor(
                executor,
                synthesize_speech,
                text_to_speak,
                speaker_id,
                temperature,
                lang_code
            )

            default_speaker_id = speaker_id

        # Save and return audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            torchaudio.save(tmp_file_path, torch.tensor(wav_array).unsqueeze(0), sample_rate)

            if not os.path.exists(tmp_file_path):
                return web.json_response({"error": "Failed to generate audio"}, status=500)

            response = web.FileResponse(
                tmp_file_path,
                headers=[('Content-Disposition', 'attachment; filename="speech.wav"')]
            )

            return response

    except Exception as e:
        logger.error(f"Error in handle_speech_request: {e}", exc_info=True)
        return web.json_response({"error": f"An error occurred: {str(e)}"}, status=500)


async def handle_speakers_list(request):
    """Handle GET /v1/speakers endpoint to list all available speakers."""
    try:
        speakers = list_available_speakers()

        response_data = {
            'speakers': speakers,
            'total': len(speakers)
        }

        return web.json_response(response_data)

    except Exception as e:
        logger.error(f"Error listing speakers: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def main():
    app = web.Application()
    app.router.add_post('/v1/audio/speech', handle_speech_request)
    app.router.add_get('/v1/speakers', handle_speakers_list)

    runner = web.AppRunner(app)
    await runner.setup()
    port = 8088
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"Tagged TTS Server started on port {port}")
    print(f"Supports tags: [silence Ns], [soundtrack Ns], [soundtrack], [speaker_id]")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("Server stopped.")
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
