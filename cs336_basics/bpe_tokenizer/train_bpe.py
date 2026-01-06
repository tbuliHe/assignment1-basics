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
    # Pattern for pre-tokenization (GPT-2)
    pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokens = re.findall(pat, text)
    
    # Initialize vocabulary: map word (tuple of bytes) -> count
    word_freqs = Counter()
    for token in tokens:
        word_bytes = tuple(bytes([b]) for b in token.encode('utf-8'))
        word_freqs[word_bytes] += 1
        
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # Optimization: Keep track of pairs and which words contain them
    # Convert word_freqs to indexable lists
    # Sort for deterministic behavior
    sorted_words = sorted(word_freqs.items(), key=lambda x: x[0])
    
    # words_list: Mutable list of current word tokenization (list of bytes)
    words_list = [list(w) for w, _ in sorted_words]
    # counts_list: Frequencies
    counts_list = [c for _, c in sorted_words]
    
    # stats: pair -> frequency
    stats = defaultdict(int)
    # indices: pair -> set of word indices
    indices = defaultdict(set)
    
    # Build initial stats
    for i, word in enumerate(words_list):
        freq = counts_list[i]
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            stats[pair] += freq
            indices[pair].add(i)
            
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    for _ in range(num_merges):
        if not stats:
            break
            
        # Find best pair. 
        # Tie-breaking: Frequency desc, then Pair asc (lexicographical)
        best_pair = min(stats.keys(), key=lambda p: (-stats[p], p))
        
        # Add to merges
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # Update words containing the pair
        current_indices = list(indices[best_pair])
        
        # We process each word that *might* have the pair
        for idx in current_indices:
            word = words_list[idx]
            
            # Check if pair is actually in word (handling lazy updates)
            # We can scan or try to merge.
            
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
                # We do NOT remove idx from indices[p] lazily to save time.
                # However we should check if stats[p] == 0 to keep stats clean
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
        del indices[best_pair] # Explicitly remove best_pair index, though stats[best_pair] should be gone.
        
    # Add special tokens
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1
        
    return vocab, merges
