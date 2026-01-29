from utils.logger import setup_logger

logger = setup_logger(__file__)


class UnifiedSpeakerRegistry:
    """
    Unified registry combining built-in and custom speakers.
    Provides single interface for speaker lookup across all sources.
    """

    def __init__(self, xtts_model, custom_speaker_manager):
        """
        Initialize the unified speaker registry.

        Args:
            xtts_model: The XTTS model instance with speaker_manager
            custom_speaker_manager: CustomSpeakerManager instance
        """
        self.xtts_model = xtts_model
        self.custom_manager = custom_speaker_manager
        self.registry = {}

    def build_registry(self):
        """
        Build unified registry from both built-in and custom speakers.

        Built-in speakers are added first, then custom speakers.
        If a custom speaker has the same ID as a built-in speaker,
        the custom speaker takes precedence.

        Returns:
            dict: The unified speaker registry
        """
        logger.info("Building unified speaker registry...")

        # Add built-in speakers first
        builtin_count = 0
        if self.xtts_model.speaker_manager is not None:
            for speaker_id, embeddings in self.xtts_model.speaker_manager.speakers.items():
                self.registry[speaker_id] = {
                    'id': speaker_id,
                    'source': 'builtin',
                    'embeddings': embeddings,
                    'cached': True
                }
                builtin_count += 1

        logger.info(f"Added {builtin_count} built-in speakers")

        # Add custom speakers (may override built-in speakers with same ID)
        custom_count = 0
        for speaker_id, speaker_data in self.custom_manager.speakers.items():
            if speaker_id in self.registry:
                logger.warning(f"Custom speaker '{speaker_id}' overrides built-in speaker")

            self.registry[speaker_id] = {
                'id': speaker_id,
                'source': 'custom',
                'embeddings': {
                    'gpt_cond_latent': speaker_data['gpt_cond_latent'],
                    'speaker_embedding': speaker_data['speaker_embedding']
                },
                'source_path': speaker_data.get('source_path', ''),
                'cache_path': speaker_data.get('cache_path', ''),
                'cached': speaker_data.get('cached', True),
                'duration': speaker_data.get('duration', 0.0)
            }
            custom_count += 1

        logger.info(f"Added {custom_count} custom speakers")
        logger.info(f"Total speakers in registry: {len(self.registry)}")

        return self.registry

    def get_speaker(self, speaker_id):
        """
        Get speaker embeddings by ID.

        Args:
            speaker_id: Speaker identifier

        Returns:
            tuple: (gpt_cond_latent, speaker_embedding) or (None, None) if not found
        """
        if speaker_id not in self.registry:
            logger.warning(f"Speaker not found: {speaker_id}")
            return None, None

        speaker = self.registry[speaker_id]
        embeddings = speaker['embeddings']

        # Handle both dict and direct tensor storage formats
        if isinstance(embeddings, dict):
            if 'gpt_cond_latent' in embeddings and 'speaker_embedding' in embeddings:
                return embeddings['gpt_cond_latent'], embeddings['speaker_embedding']
            else:
                # Built-in speakers use .values() to get tensors
                return tuple(embeddings.values())
        else:
            logger.error(f"Unexpected embeddings format for speaker: {speaker_id}")
            return None, None

    def speaker_exists(self, speaker_id):
        """
        Check if a speaker exists in the registry.

        Args:
            speaker_id: Speaker identifier

        Returns:
            bool: True if speaker exists
        """
        return speaker_id in self.registry

    def list_all_speakers(self):
        """
        List all speakers with metadata.

        Returns:
            list: List of speaker info dictionaries
        """
        speakers_list = []

        for speaker_id, speaker_data in self.registry.items():
            speaker_info = {
                'id': speaker_id,
                'source': speaker_data['source'],
                'cached': speaker_data.get('cached', True)
            }

            # Add optional fields for custom speakers
            if speaker_data['source'] == 'custom':
                speaker_info['source_path'] = speaker_data.get('source_path', '')
                speaker_info['duration'] = speaker_data.get('duration', 0.0)

            speakers_list.append(speaker_info)

        # Sort by source (builtin first) then by ID
        speakers_list.sort(key=lambda x: (x['source'] != 'builtin', x['id']))

        return speakers_list

    def get_speaker_count(self):
        """
        Get total number of speakers in registry.

        Returns:
            dict: Counts by source type
        """
        builtin = sum(1 for s in self.registry.values() if s['source'] == 'builtin')
        custom = sum(1 for s in self.registry.values() if s['source'] == 'custom')

        return {
            'total': len(self.registry),
            'builtin': builtin,
            'custom': custom
        }
