#!/usr/bin/env python3
"""
Tagged TTS Script for auralis.openai

Parses text files with voice/silence/soundtrack tags and generates audio
via the auralis.openai server.

Tag Format:
    [voice_id]      - Switch voice for subsequent text (e.g., [storyteller_1])
    [silence Ns]    - Add N seconds of silence (e.g., [silence 2s])
    [soundtrack Ns] - Add N seconds of soundtrack with fade-out (e.g., [soundtrack 30s])
    [soundtrack]    - Add default 10 seconds of soundtrack

Usage:
    python examples/tagged_tts.py input.txt -o output.mp3 \\
        --server http://127.0.0.1:8000 \\
        --soundtrack-dir ./soundtracks \\
        --default-voice storyteller_1
"""

import argparse
import io
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import requests
import soundfile as sf
import librosa


def denoise_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    noise_reduce_margin: float = 1.0,
    noise_reduce_frames: int = 25
) -> np.ndarray:
    """Apply spectral gating noise reduction to audio.

    Args:
        audio: Input audio as numpy array
        sample_rate: Sample rate of the audio
        noise_reduce_margin: Margin for noise reduction (higher = more aggressive)
        noise_reduce_frames: Number of lowest-energy frames to use for noise profile

    Returns:
        Denoised audio as numpy array
    """
    if len(audio) == 0:
        return audio

    # Ensure float32
    audio = audio.astype(np.float32)

    # Compute STFT
    D = librosa.stft(audio)
    mag, phase = librosa.magphase(D)

    # Estimate noise profile from lowest energy frames
    noise_profile = np.mean(np.sort(mag, axis=1)[:, :noise_reduce_frames], axis=1)
    noise_profile = noise_profile[:, None]

    # Create soft mask
    mask = (mag - noise_profile * noise_reduce_margin).clip(min=0)
    mask = mask / (mask + noise_profile + 1e-10)

    # Apply mask and reconstruct
    denoised = librosa.istft(mask * mag * phase, length=len(audio))

    return denoised.astype(np.float32)


# Tag regex pattern
# Matches: [silence Ns], [soundtrack Ns], [soundtrack], [voice_id]
ALL_TAGS = re.compile(
    r'\[silence\s+(\d+)s?\]|'           # [silence Ns]
    r'\[soundtrack(?:\s+(\d+)s?)?\]|'   # [soundtrack] or [soundtrack Ns]
    r'\[([a-zA-Z_][a-zA-Z0-9_]*)\]',    # [voice_id]
    re.IGNORECASE
)

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SOUNDTRACK_DURATION = 10.0
DEFAULT_FADE_OUT_DURATION = 5.0


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
    fade_out: float = DEFAULT_FADE_OUT_DURATION  # last N seconds


Chunk = Union[TextChunk, SilenceChunk, SoundtrackChunk]


def load_text_file(file_path: str) -> str:
    """Load text from a file, handling HTML if needed.

    Args:
        file_path: Path to the input file

    Returns:
        Extracted text content
    """
    path = Path(file_path)
    content = path.read_text(encoding='utf-8')

    # Check if it's HTML
    if path.suffix.lower() in ['.html', '.htm'] or content.strip().startswith('<'):
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()

            # Get text
            text = soup.get_text(separator='\n')

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return text
        except ImportError:
            print("Warning: beautifulsoup4 not installed, treating HTML as plain text")
            return content

    return content


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

    # Quote characters to strip (common quotation marks)
    QUOTE_CHARS = ['"', "'", '"', '"', "'", "'", '"']

    for match in ALL_TAGS.finditer(text):
        # Get text before this tag
        text_before = text[last_end:match.start()].strip()
        if text_before:
            # Clean up quotation marks that might wrap the dialogue
            for q in QUOTE_CHARS:
                text_before = text_before.strip(q)
            text_before = text_before.strip()
            if text_before:
                chunks.append(TextChunk(voice=current_voice, text=text_before))

        # Determine tag type
        silence_seconds = match.group(1)
        soundtrack_seconds = match.group(2)
        voice_id = match.group(3)

        if silence_seconds is not None:
            # [silence Ns] tag
            chunks.append(SilenceChunk(duration=float(silence_seconds)))
        elif match.group(0).lower().startswith('[soundtrack'):
            # [soundtrack] or [soundtrack Ns] tag
            duration = float(soundtrack_seconds) if soundtrack_seconds else DEFAULT_SOUNDTRACK_DURATION
            chunks.append(SoundtrackChunk(duration=duration))
        elif voice_id is not None:
            # [voice_id] tag - switch voice for subsequent text
            current_voice = voice_id

        last_end = match.end()

    # Get any remaining text after the last tag
    text_after = text[last_end:].strip()
    if text_after:
        # Clean up quotation marks
        for q in QUOTE_CHARS:
            text_after = text_after.strip(q)
        text_after = text_after.strip()
        if text_after:
            chunks.append(TextChunk(voice=current_voice, text=text_after))

    return chunks


def list_server_voices(server_url: str) -> List[dict]:
    """Fetch available voices from the server.

    Args:
        server_url: Base URL of the auralis.openai server

    Returns:
        List of voice dictionaries with id, aliases, language, speed
    """
    url = f"{server_url.rstrip('/')}/v1/voices"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('voices', [])
    except requests.RequestException as e:
        print(f"Error fetching voices from server: {e}")
        return []


def is_silence_or_empty(audio: np.ndarray, threshold: float = 0.001, min_duration: float = 0.5) -> bool:
    """Check if audio is essentially silence or too short.

    Args:
        audio: Audio data as numpy array
        threshold: Maximum absolute amplitude to consider as silence
        min_duration: Minimum duration in seconds to be considered valid

    Returns:
        True if audio is silence or too short
    """
    if audio is None or len(audio) == 0:
        return True
    if len(audio) < int(min_duration * DEFAULT_SAMPLE_RATE):
        return True
    if np.abs(audio).max() < threshold:
        return True
    return False


def process_text_chunk(chunk: TextChunk, server_url: str, sample_rate: int = DEFAULT_SAMPLE_RATE,
                       language: str = "auto", max_retries: int = 5, retry_delay: float = 1.0) -> np.ndarray:
    """Generate audio for a text chunk via the TTS server with retry on empty/silence.

    Args:
        chunk: TextChunk with voice and text
        server_url: Base URL of the auralis.openai server
        sample_rate: Expected sample rate of output
        language: Language code (default: auto-detect)
        max_retries: Maximum number of retries on empty/silent audio
        retry_delay: Delay between retries in seconds

    Returns:
        Audio as numpy array
    """
    url = f"{server_url.rstrip('/')}/v1/audio/speech"

    payload = {
        "input": chunk.text,
        "model": "xtts",
        "voice": chunk.voice,
        "language": language,
        "response_format": "wav",
        "speed": 1.0
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            # Read WAV data from response
            audio_data, sr = sf.read(io.BytesIO(response.content))

            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)

            audio_data = audio_data.astype(np.float32)

            # Check if audio is valid (not silence or too short)
            if is_silence_or_empty(audio_data):
                if attempt < max_retries - 1:
                    print(f"  Warning: Empty/silent audio received for '{chunk.voice}', retrying ({attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"  Error: Empty/silent audio after {max_retries} attempts for voice '{chunk.voice}'")
                    # Return short silence on error
                    return np.zeros(int(0.5 * sample_rate), dtype=np.float32)

            # Apply short fade-in/fade-out to eliminate clicks (20ms)
            fade_samples = int(0.02 * sample_rate)
            if len(audio_data) > fade_samples * 2:
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                audio_data[:fade_samples] *= fade_in
                audio_data[-fade_samples:] *= fade_out

            # Add 0.5 second silence at each end
            silence_half = np.zeros(sample_rate // 2, dtype=np.float32)
            audio_data = np.concatenate([silence_half, audio_data, silence_half])

            if attempt > 0:
                print(f"  Success on retry {attempt + 1} for voice '{chunk.voice}'")

            return audio_data

        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"  Request error for '{chunk.voice}': {e}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(retry_delay)
                continue
            print(f"  Error generating speech for voice '{chunk.voice}': {e}")
            if hasattr(e, 'response') and e.response is not None:
                error_text = e.response.text[:500]
                print(f"  Server response: {error_text}")
                # Check if it's a JSON error response
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        print(f"  Error details: {error_json['error']}")
                except:
                    pass

    # Return short silence on error
    return np.zeros(int(0.5 * sample_rate), dtype=np.float32)


def process_silence_chunk(chunk: SilenceChunk, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Generate silence audio.

    Args:
        chunk: SilenceChunk with duration
        sample_rate: Sample rate for the audio

    Returns:
        Zeros array of specified duration
    """
    n_samples = int(chunk.duration * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def process_soundtrack_chunk(
    chunk: SoundtrackChunk,
    soundtrack_dir: Optional[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Load and process a soundtrack chunk.

    Args:
        chunk: SoundtrackChunk with duration and fade_out settings
        soundtrack_dir: Directory containing soundtrack files
        sample_rate: Target sample rate

    Returns:
        Audio array with fade-out applied
    """
    if not soundtrack_dir or not os.path.isdir(soundtrack_dir):
        print(f"Warning: Soundtrack directory not found or not specified, using silence")
        return np.zeros(int(chunk.duration * sample_rate), dtype=np.float32)

    # Find audio files in soundtrack directory
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    soundtrack_files = [
        f for f in Path(soundtrack_dir).iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    if not soundtrack_files:
        print(f"Warning: No audio files found in {soundtrack_dir}, using silence")
        return np.zeros(int(chunk.duration * sample_rate), dtype=np.float32)

    # Pick a random soundtrack
    soundtrack_file = random.choice(soundtrack_files)
    print(f"Using soundtrack: {soundtrack_file.name}")

    try:
        # Load the soundtrack
        audio_data, sr = sf.read(soundtrack_file)

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)

        audio_data = audio_data.astype(np.float32)

        # Calculate target length
        target_samples = int(chunk.duration * sample_rate)

        # Loop or trim to match duration
        if len(audio_data) < target_samples:
            # Loop the audio
            repeats = (target_samples // len(audio_data)) + 1
            audio_data = np.tile(audio_data, repeats)

        # Trim to exact duration
        audio_data = audio_data[:target_samples]

        # Apply short fade-in to eliminate clicks (20ms)
        fade_in_samples = int(0.02 * sample_rate)
        if fade_in_samples < len(audio_data):
            fade_in = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
            audio_data[:fade_in_samples] *= fade_in

        # Apply fade-out
        fade_samples = int(chunk.fade_out * sample_rate)
        if fade_samples > 0 and fade_samples < len(audio_data):
            fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            audio_data[-fade_samples:] *= fade_curve

        return audio_data

    except Exception as e:
        print(f"Error loading soundtrack {soundtrack_file}: {e}")
        return np.zeros(int(chunk.duration * sample_rate), dtype=np.float32)


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Save audio array to file.

    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate of the audio
    """
    path = Path(output_path)
    suffix = path.suffix.lower()

    # Ensure audio is in valid range
    audio = np.clip(audio, -1.0, 1.0)

    if suffix == '.mp3':
        # Use soundfile for MP3 (requires libsndfile with MP3 support)
        # Fall back to writing WAV and converting
        try:
            sf.write(output_path, audio, sample_rate)
        except Exception:
            # Fall back to WAV if MP3 not supported
            wav_path = path.with_suffix('.wav')
            sf.write(str(wav_path), audio, sample_rate)
            print(f"MP3 not supported, saved as WAV: {wav_path}")
    else:
        sf.write(output_path, audio, sample_rate)


def process_chunks(
    chunks: List[Chunk],
    server_url: str,
    soundtrack_dir: Optional[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    show_progress: bool = True,
    language: str = "auto",
    max_retries: int = 5
) -> np.ndarray:
    """Process all chunks and concatenate into final audio.

    Args:
        chunks: List of chunks to process
        server_url: TTS server URL
        soundtrack_dir: Directory with soundtrack files
        sample_rate: Target sample rate
        show_progress: Whether to show progress
        language: Language code for TTS
        max_retries: Maximum retries on empty/silent audio

    Returns:
        Concatenated audio array
    """
    audio_segments: List[np.ndarray] = []
    successful_chunks = 0
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        if show_progress:
            chunk_type = type(chunk).__name__
            if isinstance(chunk, TextChunk):
                preview = chunk.text[:50] + '...' if len(chunk.text) > 50 else chunk.text
                print(f"[{i+1}/{len(chunks)}] {chunk_type} ({chunk.voice}): {preview}")
            elif isinstance(chunk, SilenceChunk):
                print(f"[{i+1}/{len(chunks)}] {chunk_type}: {chunk.duration}s")
            elif isinstance(chunk, SoundtrackChunk):
                print(f"[{i+1}/{len(chunks)}] {chunk_type}: {chunk.duration}s (fade: {chunk.fade_out}s)")

        if isinstance(chunk, TextChunk):
            audio = process_text_chunk(chunk, server_url, sample_rate, language, max_retries)
            if np.abs(audio).sum() > 0.001:  # Check if non-silence
                successful_chunks += 1
            else:
                failed_chunks += 1
        elif isinstance(chunk, SilenceChunk):
            audio = process_silence_chunk(chunk, sample_rate)
        elif isinstance(chunk, SoundtrackChunk):
            audio = process_soundtrack_chunk(chunk, soundtrack_dir, sample_rate)
        else:
            continue

        audio_segments.append(audio)

    if show_progress:
        print(f"\nChunk processing complete: {successful_chunks} successful, {failed_chunks} failed")

    if not audio_segments:
        return np.array([], dtype=np.float32)

    return np.concatenate(audio_segments)


def print_chunks(chunks: List[Chunk]) -> None:
    """Print parsed chunks for dry-run mode.

    Args:
        chunks: List of chunks to display
    """
    print("\nParsed chunks:")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, TextChunk):
            preview = chunk.text[:80] + '...' if len(chunk.text) > 80 else chunk.text
            preview = preview.replace('\n', ' ')
            print(f"{i+1}. TextChunk (voice={chunk.voice}):")
            print(f"   \"{preview}\"")
        elif isinstance(chunk, SilenceChunk):
            print(f"{i+1}. SilenceChunk: {chunk.duration}s")
        elif isinstance(chunk, SoundtrackChunk):
            print(f"{i+1}. SoundtrackChunk: {chunk.duration}s (fade-out: {chunk.fade_out}s)")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate audio from tagged text files using auralis.openai server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tag Format:
  [voice_id]      Switch voice for subsequent text
  [silence Ns]    Add N seconds of silence
  [soundtrack Ns] Add N seconds of soundtrack with 5s fade-out
  [soundtrack]    Add 10 seconds of soundtrack (default)

Examples:
  %(prog)s input.txt -o output.wav --default-voice storyteller_1
  %(prog)s story.html -o audiobook.mp3 --soundtrack-dir ./music
  %(prog)s input.txt --list-voices
  %(prog)s input.txt --dry-run
        """
    )

    parser.add_argument('input', nargs='?', help='Input file (HTML or TXT)')
    parser.add_argument('-o', '--output', help='Output audio file (WAV or MP3)')
    parser.add_argument('--server', default='http://127.0.0.1:8000',
                        help='auralis.openai server URL (default: http://127.0.0.1:8000)')
    parser.add_argument('--soundtrack-dir', help='Directory containing soundtrack files')
    parser.add_argument('--default-voice', default='storyteller_1',
                        help='Default voice for text without tags (default: storyteller_1)')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f'Output sample rate (default: {DEFAULT_SAMPLE_RATE})')
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices from server and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse input and show chunks without generating audio')
    parser.add_argument('--title', help='Title for the audio (currently unused, for metadata)')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply spectral gating noise reduction to final audio')
    parser.add_argument('--language', default='auto',
                        help='Language code for TTS (default: auto-detect)')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='Maximum retries on empty/silent audio (default: 5)')

    args = parser.parse_args()

    # Handle --list-voices
    if args.list_voices:
        print(f"Fetching voices from {args.server}...")
        voices = list_server_voices(args.server)
        if voices:
            print("\nAvailable voices:")
            print("-" * 60)
            for v in voices:
                aliases = ', '.join(v.get('aliases', [])) or 'none'
                print(f"  {v['id']}")
                print(f"    Aliases: {aliases}")
                print(f"    Language: {v.get('language', 'auto')}")
                print(f"    Speed: {v.get('speed', 1.0)}")
                print()
        else:
            print("No voices found or server not available.")
        return 0

    # Validate input file
    if not args.input:
        parser.error('Input file is required (unless using --list-voices)')

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load and parse input
    print(f"Loading input: {args.input}")
    text = load_text_file(args.input)

    print(f"Parsing tags (default voice: {args.default_voice})...")
    chunks = parse_tags(text, args.default_voice)

    if not chunks:
        print("No content found in input file.")
        return 1

    print(f"Found {len(chunks)} chunk(s)")

    # Handle --dry-run
    if args.dry_run:
        print_chunks(chunks)
        return 0

    # Validate output
    if not args.output:
        parser.error('Output file (-o) is required for audio generation')

    # Generate audio
    print(f"\nGenerating audio...")
    print(f"Server: {args.server}")
    print(f"Sample rate: {args.sample_rate}")
    if args.soundtrack_dir:
        print(f"Soundtrack dir: {args.soundtrack_dir}")
    print()

    audio = process_chunks(
        chunks,
        args.server,
        args.soundtrack_dir,
        args.sample_rate,
        language=args.language,
        max_retries=args.max_retries
    )

    if len(audio) == 0:
        print("Error: No audio generated")
        return 1

    # Apply denoising if requested
    if args.denoise:
        print("\nApplying noise reduction...")
        audio = denoise_audio(audio, args.sample_rate)

    # Save output
    print(f"\nSaving to {args.output}...")
    save_audio(audio, args.output, args.sample_rate)

    duration = len(audio) / args.sample_rate
    print(f"Done! Generated {duration:.1f}s of audio")

    return 0


if __name__ == '__main__':
    sys.exit(main())
