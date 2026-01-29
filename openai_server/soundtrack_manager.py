import os
import random
import numpy as np
import torch
import torchaudio

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SoundtrackManager:
    """
    Manage background soundtracks for multi-speaker synthesis.

    Loads audio files from a specified folder and provides methods to get
    random soundtracks with optional trimming and fade out.
    """

    SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg')

    def __init__(self, soundtrack_folder="./soundtracks", sample_rate=24000):
        """
        Initialize the soundtrack manager.

        Args:
            soundtrack_folder: Path to folder containing soundtrack audio files
            sample_rate: Target sample rate for audio playback (default 24000)
        """
        self.soundtrack_folder = soundtrack_folder
        self.sample_rate = sample_rate
        self.soundtracks = []

        self._scan_soundtracks()

    def _scan_soundtracks(self):
        """Scan the soundtrack folder for audio files."""
        if not os.path.isdir(self.soundtrack_folder):
            logger.warning(f"Soundtrack folder not found: {self.soundtrack_folder}")
            return

        for filename in os.listdir(self.soundtrack_folder):
            if filename.lower().endswith(self.SUPPORTED_FORMATS):
                filepath = os.path.join(self.soundtrack_folder, filename)
                if os.path.isfile(filepath):
                    self.soundtracks.append(filepath)

        logger.info(f"Found {len(self.soundtracks)} soundtracks in {self.soundtrack_folder}")

    def get_random_soundtrack(self, duration_seconds=10.0, fadeout_seconds=5.0):
        """
        Get a random soundtrack trimmed to the specified duration with fade out.

        Args:
            duration_seconds: Length of audio to return in seconds (default 10.0)
            fadeout_seconds: Duration of fade out at the end in seconds (default 5.0)

        Returns:
            numpy.ndarray: Audio array at the configured sample rate, or None if no soundtracks available
        """
        if not self.soundtracks:
            logger.warning("No soundtracks available")
            return None

        # Select a random soundtrack
        filepath = random.choice(self.soundtracks)
        logger.info(f"Selected soundtrack: {os.path.basename(filepath)}")

        try:
            # Load audio file
            audio, orig_sr = torchaudio.load(filepath)

            # Resample if necessary
            if orig_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
                audio = resampler(audio)

            # Convert to numpy and flatten if stereo
            audio_np = audio.numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)  # Convert stereo to mono

            # Trim to specified duration
            num_samples = int(duration_seconds * self.sample_rate)
            if len(audio_np) > num_samples:
                audio_np = audio_np[:num_samples]
            else:
                # If audio is shorter than requested duration, repeat it
                repetitions = int(np.ceil(num_samples / len(audio_np)))
                audio_np = np.tile(audio_np, repetitions)[:num_samples]

            # Apply fade out
            if fadeout_seconds > 0:
                fade_samples = min(int(fadeout_seconds * self.sample_rate), len(audio_np))
                if fade_samples > 0:
                    fade_curve = np.linspace(1.0, 0.0, fade_samples)
                    audio_np[-fade_samples:] *= fade_curve

            return audio_np.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading soundtrack {filepath}: {e}")
            return None

    def get_soundtrack_count(self):
        """Return the number of available soundtracks."""
        return len(self.soundtracks)

    def reload(self):
        """Rescan the soundtrack folder for new or removed files."""
        self.soundtracks = []
        self._scan_soundtracks()
