
def split_sentence(sentence, max_text_length=180, delimiters=",;-!?"):
    """
    Splits a sentence into two halves, prioritizing the delimiter closest to the middle.
    If no delimiter is found, it ensures words are not split in the middle.

    Args:
        sentence (str): The input sentence to split.
        delimiters (str): A string of delimiters to prioritize for splitting (default: ",;!?").

    Returns:
        tuple: A tuple containing the two halves of the sentence.
    """
    if len(sentence) < max_text_length:
        return [sentence]

    # Find all delimiter indices in the sentence
    delimiter_indices = [i for i, char in enumerate(sentence) if char in delimiters]

    if delimiter_indices:
        # Calculate the midpoint of the sentence
        midpoint = len(sentence) // 2

        # Find the delimiter closest to the midpoint
        closest_delimiter = min(delimiter_indices, key=lambda x: abs(x - midpoint))

        # Split at the closest delimiter
        first_half = sentence[:closest_delimiter].strip()
        second_half = sentence[closest_delimiter + 1:].strip()
    else:
        # If no delimiter, split at the nearest space (word boundary)
        midpoint = len(sentence) // 2

        # Find the nearest space (word boundary) around the midpoint
        left_space = sentence.rfind(" ", 0, midpoint)
        right_space = sentence.find(" ", midpoint)

        # Choose the closest space to the midpoint
        if left_space == -1 and right_space == -1:
            # No spaces found (single word), split at midpoint
            split_index = midpoint
        elif left_space == -1:
            # Only right space found
            split_index = right_space
        elif right_space == -1:
            # Only left space found
            split_index = left_space
        else:
            # Choose the closest space to the midpoint
            split_index = left_space if (midpoint - left_space) <= (right_space - midpoint) else right_space

        # Split the sentence into two parts
        first_half = sentence[:split_index].strip()
        second_half = sentence[split_index:].strip()

    return split_sentence(first_half, max_text_length=max_text_length) \
        + split_sentence(second_half, max_text_length=max_text_length)


def merge_sentences(sentences, min_words=12, max_chars=250):
    """
    Merge sentences to ensure each has at least min_words words while staying under max_chars.
    
    Strategy:
    1. Forward pass: merge short sentences with following sentences
    2. Backward pass: merge any remaining short sentences with previous ones
    3. Respect character limit to avoid overly long sentences
    
    Args:
        sentences: List of sentence strings
        min_words: Minimum number of words per sentence (default: 6)
        max_chars: Maximum characters per sentence (default: 250)
    
    Returns:
        List of merged sentences with at least min_words each and under max_chars
    """
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return sentences[:]
    
    def word_count(sentence):
        return len(sentence.split())
    
    # Forward pass: merge short sentences with next ones
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        j = i + 1
        
        # Keep merging with following sentences until we reach min_words or hit char limit
        while word_count(current) < min_words and j < len(sentences):
            next_merge = current + ' ' + sentences[j]
            if len(next_merge) > max_chars:
                break  # Would exceed character limit, stop merging
            current = next_merge
            j += 1
        
        merged.append(current)
        i = j
    
    # Backward pass: handle any remaining short sentences at the end
    while len(merged) > 1 and word_count(merged[-1]) < min_words:
        # Check if merging would exceed character limit
        potential_merge = merged[-2] + ' ' + merged[-1]
        if len(potential_merge) > max_chars:
            break  # Can't merge without exceeding limit, keep as is
        merged[-2] = potential_merge
        merged.pop()
    
    return merged


def merge_sentences_balanced(sentences, min_words=12, max_chars=250):
    """
    More balanced merging that considers both forward and backward options.
    
    This version tries to create more evenly sized sentences by choosing
    whether to merge forward or backward based on sentence lengths, while
    respecting the character limit.
    """
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return sentences[:]
    
    def word_count(sentence):
        return len(sentence.split())
    
    merged = sentences[:]
    changed = True
    
    while changed:
        changed = False
        i = 0
        
        while i < len(merged):
            if word_count(merged[i]) < min_words:
                if i < len(merged) - 1:  # Can merge forward
                    forward_merge = merged[i] + ' ' + merged[i + 1]
                    backward_merge = merged[i - 1] + ' ' + merged[i] if i > 0 else None
                    
                    # Choose merge direction based on character limits and preference
                    if len(forward_merge) <= max_chars and (
                        i == 0 or 
                        backward_merge is None or 
                        len(backward_merge) > max_chars or
                        word_count(merged[i + 1]) <= word_count(merged[i - 1])
                    ):
                        # Merge forward
                        merged[i] = forward_merge
                        merged.pop(i + 1)
                        changed = True
                    elif i > 0 and backward_merge and len(backward_merge) <= max_chars:
                        # Merge backward
                        merged[i - 1] = backward_merge
                        merged.pop(i)
                        changed = True
                        i -= 1
                elif i > 0:  # Can only merge backward
                    backward_merge = merged[i - 1] + ' ' + merged[i]
                    if len(backward_merge) <= max_chars:
                        merged[i - 1] = backward_merge
                        merged.pop(i)
                        changed = True
                        i -= 1
            i += 1
    
    return merged
