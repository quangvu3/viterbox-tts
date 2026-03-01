import os
import sys
import time
import random
import tempfile
import argparse
from pathlib import Path

import asyncio
from aiohttp import web

import torch
import torchaudio
import numpy as np

from underthesea import sent_tokenize

from langdetect import detect

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(APP_DIR)

import logging

# Simple logger setup
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

# Language mapping: UI names to Viterbox codes
language_dict = {
    'Tiếng Việt': 'vi',
    'Tiếng Việt (Vietnamese)': 'vi',
    'English': 'en',
    'English (US)': 'en',
    'Auto': 'auto',
}

default_language = 'vi'
language_codes = list(language_dict.values())
normalize_text_enabled = True


def parse_args():
    parser = argparse.ArgumentParser(description="Viterbox TTS Server")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization before synthesis",
    )
    return parser.parse_args()

def lang_detect(text):
    """Detect language from text."""
    try:
        lang = detect(text)
        # Map detected lang to our supported codes
        if lang in language_codes:
            return lang
        # Common mappings
        if lang.startswith('zh'):
            return 'vi'  # Default to Vietnamese for Chinese
        return 'vi'  # Default to Vietnamese
    except:
        return 'vi'

# Viterbox model
viterbox_model = None

def load_model():
    global viterbox_model
    viterbox_model = Viterbox.from_pretrained(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Viterbox model loaded successfully")

load_model()

# Speaker management
speakers_dir = f"{APP_DIR}/speakers"
default_speaker_id = "storyteller_1"

# Cache for last used speaker ID
last_used_speaker_id = default_speaker_id


def get_speaker_audio_path(speaker_id: str) -> str:
    """Get reference audio path for a speaker from speakers/ folder."""
    speaker_wav = Path(f"{speakers_dir}/{speaker_id}.wav")
    if speaker_wav.exists():
        return str(speaker_wav)

    # Also check for alternative naming patterns
    speaker_dir = Path(f"{speakers_dir}/{speaker_id}")
    if speaker_dir.exists():
        wav_files = list(speaker_dir.glob("*.wav"))
        if wav_files:
            return str(wav_files[0])

    raise ValueError(f"No reference audio found for speaker: {speaker_id}")


def list_available_speakers() -> list:
    """List all available speakers from the speakers/ folder."""
    speakers = []

    # Check for direct .wav files
    for f in Path(speakers_dir).glob("*.wav"):
        speaker_id = f.stem
        speakers.append({
            'id': speaker_id,
            'source': 'local',
            'cached': True,
            'path': str(f)
        })

    # Also check subdirectories
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


def synthesize_speech(input_text, speaker_id, temperature=0.5, language='vi'):
    """Process text and generate audio using Viterbox."""
    global viterbox_model, last_used_speaker_id, normalize_text_enabled

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
        temperature=round(random.uniform(0.45, 0.55), 4),
        cfg_weight=round(random.uniform(0.28, 0.32), 4),
        top_p=0.9,
        repetition_penalty=1.2,
        split_sentences=True,
        sentence_pause_ms=500,
        dereverberation=True,
        dereverberation_strength=0.5,
        normalize_text_enabled=normalize_text_enabled,
    )

    end = time.time()
    processing_time = end - start

    # Convert tensor to numpy
    wav_array = wav_tensor[0].cpu().numpy()

    # Calculate approximate tokens (based on audio length at 24kHz)
    num_of_tokens = int(len(wav_array) * 50 / 24000)  # Approximate
    tokens_per_second = num_of_tokens / processing_time if processing_time > 0 else 0

    logger.info(f"End processing text: {input_text[:30]}")
    message = f"💡 {tokens_per_second:.1f} tok/s • {num_of_tokens} tokens • in {processing_time:.2f} seconds"
    logger.info(message)

    # Update cached speaker
    last_used_speaker_id = speaker_id

    return (viterbox_model.sr, wav_array)


def inference(input_text, language, speaker_id=None, temperature=0.5, sentence_silence_ms=500):
    """
    Generate speech from text using Viterbox.

    Args:
        input_text: Text to synthesize
        language: Target language
        speaker_id: Speaker identifier for voice cloning
        temperature: Sampling temperature
        sentence_silence_ms: Silence between sentences (kept for API compatibility)

    Returns:
        tuple: (final_wav_array, num_of_tokens)
    """
    global viterbox_model, normalize_text_enabled

    # Bypass text with fewer than 2 alphabetic characters
    alpha_count = sum(1 for c in input_text if c.isalpha()) if input_text else 0
    if alpha_count < 2:
        logger.debug(f"Skipping text with insufficient alphabetic chars ({alpha_count}): '{input_text[:50] if input_text else input_text}'")
        return np.array([]), 0

    # Use default speaker if none specified
    if speaker_id is None:
        speaker_id = default_speaker_id

    # Detect language
    lang_code = lang_detect(input_text) if language == 'auto' else language_dict.get(language, 'vi')

    # Get reference audio path
    ref_path = get_speaker_audio_path(speaker_id)

    # Split text by sentence
    if lang_code in ["ja", "zh-cn"]:
        sentences = input_text.split("。")
    else:
        sentences = sent_tokenize(input_text)

    # Generate each sentence
    out_wavs = []
    num_of_tokens = 0

    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) == 0:
            continue

        # Detect language per sentence if auto
        sentence_lang = lang_detect(sentence) if language == 'auto' else lang_code

        logger.info(f"[{sentence_lang}] {sentence}")

        try:
            # Generate with Viterbox
            wav_tensor = viterbox_model.generate(
                text=sentence,
                language=sentence_lang,
                audio_prompt=ref_path,
                temperature=round(random.uniform(0.45, 0.55), 4),
                cfg_weight=round(random.uniform(0.28, 0.32), 4),
                top_p=0.9,
                repetition_penalty=1.2,
                split_sentences=True,
                sentence_pause_ms=sentence_silence_ms if i < len(sentences) - 1 else 0,
                dereverberation=True,
                normalize_text_enabled=normalize_text_enabled,
            )

            # Convert to numpy
            sentence_wav = wav_tensor[0].cpu().numpy()

            # Estimate token count from audio length
            tokens_for_sentence = int(len(sentence_wav) * 50 / 24000)
            num_of_tokens += tokens_for_sentence

            out_wavs.append(sentence_wav)

        except Exception as e:
            logger.error(f"Error processing sentence: {sentence} - {e}")

    # Concatenate all sentences
    if out_wavs:
        final_wav = np.concatenate(out_wavs)
    else:
        final_wav = np.array([])

    return final_wav, num_of_tokens


async def handle_speech_request(request):
    """Handles the /v1/audio/speech endpoint."""
    global last_used_speaker_id

    try:
        request_data = await request.json()
        text_to_speak = request_data.get('text')
        language = request_data.get('language', 'Tiếng Việt')
        speaker_id = request_data.get('speaker', last_used_speaker_id)

        if not text_to_speak:
            return web.json_response({"error": "Missing or empty 'text' field"}, status=400)

        # Validate speaker exists
        try:
            ref_path = get_speaker_audio_path(speaker_id)
        except ValueError:
            # Try default speaker
            try:
                ref_path = get_speaker_audio_path(default_speaker_id)
                speaker_id = default_speaker_id
            except ValueError:
                return web.json_response({"error": f"No reference audio found for any speaker"}, status=400)

        logger.info(f"Using speaker: {speaker_id}")

        # Map language to code
        lang_code = language_dict.get(language, 'vi')

        # Generate speech
        sample_rate, wav_array = synthesize_speech(
            input_text=text_to_speak,
            speaker_id=speaker_id,
            language=lang_code
        )

        # Update cached speaker
        last_used_speaker_id = speaker_id

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
    print(f"Server started on port {port}")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("Server stopped.")
        await runner.cleanup()

if __name__ == '__main__':
    args = parse_args()
    normalize_text_enabled = not args.no_normalize
    logger.info("Text normalization: %s", "enabled" if normalize_text_enabled else "disabled")
    asyncio.run(main())
