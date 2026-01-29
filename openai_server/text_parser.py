import re
from utils.logger import setup_logger

logger = setup_logger(__file__)


class TextParser:
    """
    Parse text with embedded speaker and silence tags.

    Supported formats:
    - Speaker tags: [speaker_id]
    - Silence tags: [silence 2s], [silence 0.5s]
    - Soundtrack tags: [soundtrack 10s], [soundtrack 10s fadeout:3s]

    Example:
        [main_storyteller_1] Once upon a time... [silence 1s] [hero_voice] Hello!
        [soundtrack 10s] [narrator] Once upon a time... [soundtrack 15s fadeout:3s] [hero] Hello!
    """

    # Regex pattern for silence tags: [silence 2s] or [silence 0.5s]
    SILENCE_TAG_PATTERN = r'\[silence\s+(\d+(?:\.\d+)?)\s*s\]'

    # Regex pattern for soundtrack tags: [soundtrack 10s] or [soundtrack 10s fadeout:3s]
    SOUNDTRACK_TAG_PATTERN = r'\[soundtrack(?:\s+(\d+(?:\.\d+)?)\s*s)?(?:\s+fadeout:(\d+(?:\.\d+)?)\s*s)?\]'

    # Regex pattern for any tag: [something]
    TAG_PATTERN = r'\[([^\]]+?)\]'

    def __init__(self, speaker_registry):
        """
        Initialize the text parser.

        Args:
            speaker_registry: UnifiedSpeakerRegistry instance for speaker validation
        """
        self.speaker_registry = speaker_registry

    def parse_text(self, text, default_speaker=None):
        """
        Parse text into segments with speaker/silence tags.

        Args:
            text: Input text with embedded tags
            default_speaker: Default speaker to use if no initial tag (optional)

        Returns:
            list: List of segment dictionaries
                [{type: 'speech', speaker_id: str, text: str},
                 {type: 'silence', duration: float}]

        Raises:
            ValueError: If text has no speaker tags and no default speaker
        """
        segments = []
        current_speaker = default_speaker
        current_pos = 0

        # Find all tags (both speaker and silence)
        tag_matches = list(re.finditer(self.TAG_PATTERN, text))

        if not tag_matches:
            # No tags found - use entire text with default speaker
            if not default_speaker:
                raise ValueError("No speaker tags found and no default speaker provided")

            text_content = text.strip()
            if text_content:
                segments.append({
                    'type': 'speech',
                    'speaker_id': default_speaker,
                    'text': text_content
                })
            return segments

        for match in tag_matches:
            tag_start = match.start()
            tag_end = match.end()
            tag_content = match.group(1)

            # Extract text before this tag (if any)
            text_before = text[current_pos:tag_start].strip()
            if text_before and current_speaker:
                segments.append({
                    'type': 'speech',
                    'speaker_id': current_speaker,
                    'text': text_before
                })

            # Parse the tag
            silence_match = re.match(r'silence\s+(\d+(?:\.\d+)?)\s*s', tag_content)
            if silence_match:
                # It's a silence tag
                duration = float(silence_match.group(1))
                if duration > 10.0:
                    logger.warning(f"Very long silence requested: {duration}s")
                segments.append({
                    'type': 'silence',
                    'duration': duration
                })
            else:
                # Check if it's a soundtrack tag
                soundtrack_match = re.match(self.SOUNDTRACK_TAG_PATTERN, f'[{tag_content}]')
                if soundtrack_match:
                    # It's a soundtrack tag
                    duration = float(soundtrack_match.group(1)) if soundtrack_match.group(1) else 10.0
                    fadeout = float(soundtrack_match.group(2)) if soundtrack_match.group(2) else 5.0
                    segments.append({
                        'type': 'soundtrack',
                        'duration': duration,
                        'fadeout': fadeout
                    })
                else:
                    # It's a speaker tag
                    current_speaker = tag_content.strip()

            current_pos = tag_end

        # Handle remaining text after last tag
        text_after = text[current_pos:].strip()
        if text_after and current_speaker:
            segments.append({
                'type': 'speech',
                'speaker_id': current_speaker,
                'text': text_after
            })

        # Check if we have at least one speech segment
        if not any(seg['type'] == 'speech' for seg in segments):
            if default_speaker:
                # No speech segments but have default speaker, treat entire text as speech
                text_content = re.sub(self.TAG_PATTERN, '', text).strip()
                if text_content:
                    segments = [{
                        'type': 'speech',
                        'speaker_id': default_speaker,
                        'text': text_content
                    }]
            else:
                raise ValueError("No speech segments found and no default speaker provided")

        return segments

    def validate_speakers(self, segments):
        """
        Validate that all speaker IDs in segments exist in registry.

        Args:
            segments: List of segments from parse_text()

        Raises:
            ValueError: If any speaker ID is not found
        """
        missing_speakers = []

        for segment in segments:
            if segment['type'] == 'speech':
                speaker_id = segment['speaker_id']
                if not self.speaker_registry.speaker_exists(speaker_id):
                    missing_speakers.append(speaker_id)

        if missing_speakers:
            unique_missing = list(set(missing_speakers))
            raise ValueError(f"Speaker(s) not found: {', '.join(unique_missing)}")

    def has_tags(self, text):
        """
        Check if text contains any tags.

        Args:
            text: Input text

        Returns:
            bool: True if text contains tags
        """
        return bool(re.search(self.TAG_PATTERN, text))

    def remove_all_tags(self, text):
        """
        Remove all tags from text, leaving only the text content.

        Args:
            text: Input text with tags

        Returns:
            str: Text with all tags removed
        """
        return re.sub(self.TAG_PATTERN, '', text).strip()

    def get_unique_speakers(self, segments):
        """
        Get list of unique speaker IDs from segments.

        Args:
            segments: List of segments from parse_text()

        Returns:
            list: List of unique speaker IDs
        """
        speakers = set()
        for segment in segments:
            if segment['type'] == 'speech':
                speakers.add(segment['speaker_id'])
        return sorted(list(speakers))

    def segment_stats(self, segments):
        """
        Get statistics about parsed segments.

        Args:
            segments: List of segments from parse_text()

        Returns:
            dict: Statistics dictionary
        """
        speech_count = sum(1 for s in segments if s['type'] == 'speech')
        silence_count = sum(1 for s in segments if s['type'] == 'silence')
        soundtrack_count = sum(1 for s in segments if s['type'] == 'soundtrack')
        total_silence_duration = sum(s['duration'] for s in segments if s['type'] == 'silence')
        total_soundtrack_duration = sum(s['duration'] for s in segments if s['type'] == 'soundtrack')
        unique_speakers = len(self.get_unique_speakers(segments))

        return {
            'total_segments': len(segments),
            'speech_segments': speech_count,
            'silence_segments': silence_count,
            'soundtrack_segments': soundtrack_count,
            'total_silence_duration': total_silence_duration,
            'total_soundtrack_duration': total_soundtrack_duration,
            'unique_speakers': unique_speakers
        }
