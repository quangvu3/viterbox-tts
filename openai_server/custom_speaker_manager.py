import os
import glob
from pathlib import Path
from datetime import datetime
from safetensors.torch import save_file, load_file
import torchaudio

from utils.logger import setup_logger

logger = setup_logger(__file__)


class CustomSpeakerManager:
    """
    Manages custom speakers from audio files in speakers/ directory.
    Handles loading, processing, caching, and retrieval of speaker embeddings.
    """

    def __init__(self, xtts_model, speakers_dir, cache_dir):
        """
        Initialize the custom speaker manager.

        Args:
            xtts_model: The XTTS model instance with get_conditioning_latents method
            speakers_dir: Path to directory containing audio files (e.g., ./speakers/)
            cache_dir: Path to cache directory for .safetensors files (e.g., ./cache/speakers/custom/)
        """
        self.xtts_model = xtts_model
        self.speakers_dir = speakers_dir
        self.cache_dir = cache_dir
        self.speakers = {}

        os.makedirs(cache_dir, exist_ok=True)

    def scan_and_load_speakers(self):
        """
        Scan speakers directory and load/cache all speaker embeddings.

        This method:
        1. Finds all .wav and .mp3 files in speakers_dir
        2. For each file, checks if cache exists and is valid
        3. If not cached or outdated, processes audio to embeddings
        4. Loads embeddings into memory

        Returns:
            dict: Dictionary of speaker_id -> embeddings
        """
        if not os.path.exists(self.speakers_dir):
            logger.warning(f"Speakers directory not found: {self.speakers_dir}")
            return self.speakers

        # Find all audio files
        audio_patterns = ['*.wav', '*.mp3', '*.flac', '*.ogg']
        audio_files = []
        for pattern in audio_patterns:
            audio_files.extend(glob.glob(os.path.join(self.speakers_dir, pattern)))

        if not audio_files:
            logger.info(f"No audio files found in {self.speakers_dir}")
            return self.speakers

        logger.info(f"Found {len(audio_files)} audio files in {self.speakers_dir}")

        # Process each audio file
        for audio_path in audio_files:
            speaker_id = Path(audio_path).stem

            try:
                # Check if cache exists and is valid
                if self._is_cache_valid(audio_path, speaker_id):
                    logger.info(f"Loading cached speaker: {speaker_id}")
                    self._load_cached_speaker(speaker_id, audio_path)
                else:
                    logger.info(f"Processing speaker: {speaker_id}")
                    self.process_and_cache_speaker(audio_path, speaker_id)

            except Exception as e:
                logger.error(f"Failed to process speaker {speaker_id}: {e}")
                continue

        logger.info(f"Loaded {len(self.speakers)} custom speakers")
        return self.speakers

    def _is_cache_valid(self, audio_path, speaker_id):
        """
        Check if cached embeddings exist and are newer than source audio.

        Args:
            audio_path: Path to source audio file
            speaker_id: Speaker identifier

        Returns:
            bool: True if cache is valid, False otherwise
        """
        cache_path = os.path.join(self.cache_dir, f"{speaker_id}.safetensors")

        if not os.path.exists(cache_path):
            return False

        # Check modification times
        audio_mtime = os.path.getmtime(audio_path)
        cache_mtime = os.path.getmtime(cache_path)

        return cache_mtime > audio_mtime

    def _load_cached_speaker(self, speaker_id, audio_path):
        """
        Load speaker embeddings from cache.

        Args:
            speaker_id: Speaker identifier
            audio_path: Path to source audio (for metadata)
        """
        cache_path = os.path.join(self.cache_dir, f"{speaker_id}.safetensors")

        # Load the safetensors file
        tensors = load_file(cache_path)

        # Extract embeddings (metadata is stored as strings in safetensors)
        self.speakers[speaker_id] = {
            'gpt_cond_latent': tensors['gpt_cond_latent'],
            'speaker_embedding': tensors['speaker_embedding'],
            'source_path': audio_path,
            'cache_path': cache_path,
            'cached': True
        }

    def process_and_cache_speaker(self, audio_path, speaker_id):
        """
        Process audio file to create embeddings and cache them.

        Args:
            audio_path: Path to audio file
            speaker_id: Speaker identifier

        Returns:
            dict: Speaker embeddings dictionary
        """
        logger.info(f"Processing audio: {audio_path}")

        # Get audio info for metadata
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            logger.warning(f"Could not get audio info for {audio_path}: {e}")
            duration = 0.0

        # Process audio to get conditioning latents
        gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
            audio_path=audio_path,
            gpt_cond_len=self.xtts_model.config.gpt_cond_len,
            max_ref_length=self.xtts_model.config.max_ref_len,
            sound_norm_refs=self.xtts_model.config.sound_norm_refs,
        )

        # Prepare cache data - ensure tensors are contiguous for safetensors
        cache_data = {
            'gpt_cond_latent': gpt_cond_latent.contiguous(),
            'speaker_embedding': speaker_embedding.contiguous(),
        }

        # Save to cache
        cache_path = os.path.join(self.cache_dir, f"{speaker_id}.safetensors")
        save_file(cache_data, cache_path)

        logger.info(f"Cached speaker embeddings to: {cache_path}")

        # Store in memory
        self.speakers[speaker_id] = {
            'gpt_cond_latent': gpt_cond_latent,
            'speaker_embedding': speaker_embedding,
            'source_path': audio_path,
            'cache_path': cache_path,
            'cached': True,
            'duration': duration
        }

        return self.speakers[speaker_id]

    def get_speaker_embeddings(self, speaker_id):
        """
        Get embeddings for a speaker ID.

        Args:
            speaker_id: Speaker identifier

        Returns:
            tuple: (gpt_cond_latent, speaker_embedding) or (None, None) if not found
        """
        if speaker_id not in self.speakers:
            return None, None

        speaker = self.speakers[speaker_id]
        return speaker['gpt_cond_latent'], speaker['speaker_embedding']

    def list_speakers(self):
        """
        Return list of all available custom speakers with metadata.

        Returns:
            list: List of speaker info dictionaries
        """
        speakers_list = []
        for speaker_id, speaker_data in self.speakers.items():
            speakers_list.append({
                'id': speaker_id,
                'source': 'custom',
                'source_path': speaker_data.get('source_path', ''),
                'cache_path': speaker_data.get('cache_path', ''),
                'cached': speaker_data.get('cached', True),
                'duration': speaker_data.get('duration', 0.0)
            })
        return speakers_list

    def speaker_exists(self, speaker_id):
        """
        Check if a speaker exists.

        Args:
            speaker_id: Speaker identifier

        Returns:
            bool: True if speaker exists
        """
        return speaker_id in self.speakers
