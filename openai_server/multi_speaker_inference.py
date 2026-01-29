import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__file__)


class MultiSpeakerInference:
    """
    Handle multi-speaker inference with automatic speaker switching and silence insertion.
    """

    def __init__(self, xtts_model, speaker_registry, inference_fn, soundtrack_manager=None, speaker_stats_tracker=None):
        """
        Initialize the multi-speaker inference engine.

        Args:
            xtts_model: The XTTS model instance
            speaker_registry: UnifiedSpeakerRegistry instance
            inference_fn: The inference function to use for TTS generation
            soundtrack_manager: Optional SoundtrackManager instance for background music
            speaker_stats_tracker: Optional SpeakerStatsTracker for per-speaker stats
        """
        self.xtts_model = xtts_model
        self.speaker_registry = speaker_registry
        self.inference_fn = inference_fn
        self.soundtrack_manager = soundtrack_manager
        self.speaker_stats_tracker = speaker_stats_tracker

    def synthesize_segments(
        self,
        segments,
        language='Auto',
        temperature=0.3,
        top_p=0.85,
        top_k=50,
        repetition_penalty=29.0,
        sentence_silence_ms=500
    ):
        """
        Synthesize audio from parsed segments with multiple speakers.
        Automatically adds 1 second of silence between different speakers.

        Args:
            segments: List of parsed segments from TextParser
            language: Target language
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
            sentence_silence_ms: Silence to add between sentences (milliseconds)

        Returns:
            tuple: (final_wav_array, total_tokens)
        """
        out_wavs = []
        total_tokens = 0
        previous_speaker = None

        logger.info(f"Synthesizing {len(segments)} segments")

        for i, segment in enumerate(segments):
            if segment['type'] == 'silence':
                # Generate silence
                duration = segment['duration']
                logger.info(f"Segment {i+1}: Silence ({duration}s)")
                silence_wav = self._generate_silence(duration)
                out_wavs.append(silence_wav)

            elif segment['type'] == 'soundtrack':
                # Play background soundtrack
                duration = segment['duration']
                fadeout = segment.get('fadeout', 5.0)
                logger.info(f"Segment {i+1}: Soundtrack ({duration}s, fadeout: {fadeout}s)")
                soundtrack_wav = self._process_soundtrack(duration, fadeout)
                if soundtrack_wav is not None:
                    out_wavs.append(soundtrack_wav)
                else:
                    # Fallback to silence if no soundtrack available
                    silence_wav = self._generate_silence(duration)
                    out_wavs.append(silence_wav)

            elif segment['type'] == 'speech':
                # Get speaker embeddings
                speaker_id = segment['speaker_id']
                text = segment['text']

                # Add 1 second silence between different speakers
                if previous_speaker is not None and previous_speaker != speaker_id:
                    logger.info(f"Speaker change detected: [{previous_speaker}] -> [{speaker_id}], adding 1s silence")
                    silence_wav = self._generate_silence(1.0)
                    out_wavs.append(silence_wav)

                logger.info(f"Segment {i+1}: Speaker [{speaker_id}] - {text[:50]}...")

                # Get speaker embeddings from registry
                gpt_cond_latent, speaker_embedding = self.speaker_registry.get_speaker(speaker_id)

                if gpt_cond_latent is None or speaker_embedding is None:
                    logger.error(f"Failed to get embeddings for speaker: {speaker_id}")
                    continue

                try:
                    # Call the inference function
                    wav, tokens = self.inference_fn(
                        input_text=text,
                        language=language,
                        speaker_id=speaker_id,  # Pass speaker_id for stats tracking
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        sentence_silence_ms=sentence_silence_ms
                    )

                    # Record stats for this speaker
                    if self.speaker_stats_tracker and speaker_id:
                        # Handle the case where audio might be a torch tensor
                        audio_for_stats = wav
                        if hasattr(audio_for_stats, 'cpu'):
                            audio_for_stats = audio_for_stats.cpu().numpy()
                        if isinstance(audio_for_stats, np.ndarray) and audio_for_stats.ndim > 1:
                            audio_for_stats = audio_for_stats.squeeze()
                        audio_samples = len(audio_for_stats)

                        # Calculate word and char counts
                        word_count = len(text.split())
                        char_count = len(text)

                        # Detect language from text if needed
                        from langdetect import detect
                        try:
                            lang = detect(text)
                            # Map to our language codes
                            if lang == 'zh-tw':
                                lang = 'zh-cn'
                            elif lang not in ['en', 'vi', 'ja', 'zh-cn', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'hu']:
                                lang = 'en'
                        except:
                            lang = 'en'

                        # Record the generation
                        self.speaker_stats_tracker.record_generation(
                            speaker_id=speaker_id,
                            language=lang,
                            word_count=word_count,
                            char_count=char_count,
                            audio_samples=audio_samples,
                            sample_rate=self.xtts_model.config.audio.sample_rate
                        )

                    out_wavs.append(wav)
                    total_tokens += tokens
                    previous_speaker = speaker_id  # Update previous speaker

                except Exception as e:
                    logger.error(f"Error synthesizing segment {i+1}: {e}")
                    continue

        # Concatenate all audio segments
        if out_wavs:
            final_wav = np.concatenate(out_wavs)
            logger.info(f"Successfully synthesized {len(out_wavs)} segments, {total_tokens} total tokens")
        else:
            logger.warning("No audio segments generated")
            final_wav = np.array([])

        return final_wav, total_tokens

    def _generate_silence(self, duration_seconds):
        """
        Generate silence array.

        Args:
            duration_seconds: Duration of silence in seconds

        Returns:
            numpy.ndarray: Array of zeros representing silence
        """
        sample_rate = self.xtts_model.config.audio.sample_rate
        num_samples = int(duration_seconds * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def _process_soundtrack(self, duration_seconds, fadeout_seconds):
        """
        Process a soundtrack segment.

        Args:
            duration_seconds: Duration of the soundtrack in seconds
            fadeout_seconds: Duration of fade out at the end in seconds

        Returns:
            numpy.ndarray: Audio array with the soundtrack, or None if unavailable
        """
        if self.soundtrack_manager is None:
            logger.warning("Soundtrack manager not initialized")
            return None

        return self.soundtrack_manager.get_random_soundtrack(duration_seconds, fadeout_seconds)

    def estimate_duration(self, segments):
        """
        Estimate total audio duration from segments.

        This is a rough estimate based on text length and silence durations.

        Args:
            segments: List of parsed segments

        Returns:
            float: Estimated duration in seconds
        """
        total_duration = 0.0

        for segment in segments:
            if segment['type'] == 'silence':
                total_duration += segment['duration']
            elif segment['type'] == 'soundtrack':
                total_duration += segment['duration']
            elif segment['type'] == 'speech':
                # Rough estimate: ~150 words per minute for speech
                # That's 2.5 words per second, or 0.4 seconds per word
                word_count = len(segment['text'].split())
                total_duration += word_count * 0.4

        return total_duration
