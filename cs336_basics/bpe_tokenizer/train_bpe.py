from typing import Any, BinaryIO, IO
import os
import regex as re
from collections import Counter, defaultdict

def get_stats(vocab: dict[bytes, int]) -> dict[tuple[bytes, bytes], int]:
    """Given a vocabulary (word -> frequency), return frequency of pair of tokens."""
    pairs = defaultdict(int) 
    for word, freq in vocab.items():
        # word is already a tuple of bytes (tokens)
        symbols = word
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] += freq
    return pairs

def merge_vocab(pair: tuple[bytes, bytes], v_in: dict[bytes, int]) -> dict[bytes, int]:
    """Merge a pair in all words in the given vocabulary."""
    v_out = {}
    p0, p1 = pair
    new_token = p0 + p1
    
    for word, freq in v_in.items():
        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                new_word_list.append(new_token)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        v_out[tuple(new_word_list)] = freq
    return v_out

def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer on a given corpus.
    
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: The trained tokenizer vocabulary and merges.
    """
    # 1. Byte Encoder (for GPT-2 style string representation)
    byte_encoder = bytes_to_unicode()
    
    # 2. Read text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 3. Handle Special Tokens (Split text to avoid tokenizing special tokens)
    if special_tokens:
        # Sort by length desc to ensure longest match
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        # Verify that special tokens are unique, though usage here doesn't strictly require it
        # Escape for regex
        escaped_special = [re.escape(s) for s in sorted_special_tokens]
        special_pattern = "(" + "|".join(escaped_special) + ")"
        parts = re.split(special_pattern, text)
    else:
        parts = [text]
        sorted_special_tokens = [] # for safety check later

    # Pattern for pre-tokenization (GPT-2)
    pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # 4. Build frequency counts
    # Map: tuple[str, ...] -> count. The strings are the mapped unicode chars.
    word_freqs = Counter()
    
    for part in parts:
        # If this part is a special token, skip it for BPE training
        if part in special_tokens:
            continue
            
        # Apply GPT-2 regex
        tokens = re.findall(pat, part)
        
        for token in tokens:
            # Convert token string to bytes, then map each byte to unicode char
            token_bytes = token.encode('utf-8')
            # Create a tuple of the mapped characters
            word_chars = tuple(byte_encoder[b] for b in token_bytes)
            word_freqs[word_chars] += 1
        
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # Optimization: Keep track of pairs and which words contain them
    # Sort for deterministic behavior. 
    # NOTE: The sort key is the tuple of strings. This matches GPT-2 tie-breaking rules
    # where characters are sorted by their unicode code points.
    sorted_words = sorted(word_freqs.items(), key=lambda x: x[0])
    
    # words_list: Mutable list of current word tokenization (list of strings)
    words_list = [list(w) for w, _ in sorted_words]
    # counts_list: Frequencies
    counts_list = [c for _, c in sorted_words]
    
    # stats: pair (str, str) -> frequency
    stats = defaultdict(int)
    # indices: pair (str, str) -> set of word indices
    indices = defaultdict(set)
    
    # Build initial stats
    for i, word in enumerate(words_list):
        freq = counts_list[i]
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            stats[pair] += freq
            indices[pair].add(i)
            
    merges = []
    # We will build the final vocab at the end, or track it. 
    # Let's track the merge operations.
    
    for _ in range(num_merges):
        if not stats:
            break
            
        # Find best pair. 
        # Tie-breaking: Frequency desc, then Pair asc (lexicographical)
        # Since our 'Pair' contains mapped unicode chars, this lexicographical sort 
        # aligns with GPT-2's correctness requirement (vs raw bytes)
        best_pair = min(stats.keys(), key=lambda p: (-stats[p], p))
        
        # Add to merges
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        # Update words containing the pair
        current_indices = list(indices[best_pair])
        
        # We process each word that *might* have the pair
        for idx in current_indices:
            word = words_list[idx]
            
            # Check if pair is actually in word (handling lazy updates)
            # Form new word
            new_word = []
            i = 0
            changed = False
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                    changed = True
                else:
                    new_word.append(word[i])
                    i += 1
            
            if not changed:
                continue
                
            # If changed, update stats
            freq = counts_list[idx]
            
            # 1. Remove stats for old word
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                stats[p] -= freq
                if stats[p] == 0:
                    del stats[p]

            # 2. Update word
            words_list[idx] = new_word
            
            # 3. Add stats for new word
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                stats[p] += freq
                indices[p].add(idx)
        
        # Cleanup
        if best_pair in indices:
            del indices[best_pair]
            
    # Post-processing: Convert results back to bytes
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    def decode_token(token_str: str) -> bytes:
        return bytes([byte_decoder[c] for c in token_str])
    
    final_vocab = {}
    # 0-255
    for i in range(256):
        final_vocab[i] = bytes([i])
        
    # Merges
    # The 'merges' list contains tuples of strings. 
    # We need to construct the vocab in order.
    # The trained merges imply the creation of new tokens.
    final_merges_bytes = []
    
    for i, (p0, p1) in enumerate(merges):
        # Add to vocab
        token_str = p0 + p1
        final_vocab[256 + i] = decode_token(token_str)
        
        # Add to list of merges (in bytes)
        final_merges_bytes.append((decode_token(p0), decode_token(p1)))
        
    # Add special tokens
    base_idx = 256 + len(final_merges_bytes)
    for st in special_tokens:
        final_vocab[base_idx] = st.encode('utf-8')
        base_idx += 1
        
    return final_vocab, final_merges_bytes
