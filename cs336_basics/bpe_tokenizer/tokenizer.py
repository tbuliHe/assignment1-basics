import regex as re
from typing import Any

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize BPE Tokenizer.
        
        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID) to bytes.
            merges (list[tuple[bytes, bytes]]): BPE merges.
            special_tokens (list[str] | None): A list of string special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Invert vocab for encoding
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # Rank merges for priority (lower index = higher priority)
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        
        # Cache for BPE
        self.cache = {}
        
        # Regex pattern from GPT-2
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token: bytes) -> list[bytes]:
        """Apply BPE to a single token (byte sequence)."""
        if token in self.cache:
            return self.cache[token]
            
        word = [bytes([b]) for b in token]
        pairs = self.get_pairs(word)
        
        if not pairs:
            return word
            
        while True:
            # Find the pair with the lowest rank (earliest in merges)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                    
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
                
        self.cache[token] = word
        return word

    def get_pairs(self, word: list[bytes]):
        return set(zip(word, word[1:]))

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if not text:
            return []
            
        # Handle special tokens
        special_token_map = {t: self.token_to_id.get(t.encode('utf-8')) for t in self.special_tokens}
        
        # Create pattern to split by special tokens first
        if self.special_tokens:
            # Sort by length descending to ensure longest match is used
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(k) for k in sorted_special) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
            
        ids = []
        for part in parts:
            if part in self.special_tokens:
                part_bytes = part.encode('utf-8')
                if part_bytes in self.token_to_id:
                     ids.append(self.token_to_id[part_bytes])
                continue
            
            # Normal text
            for match in re.findall(self.pat, part):
                token_bytes = match.encode('utf-8')
                bpe_tokens = self.bpe(token_bytes)
                for bpe_token in bpe_tokens:
                    # Fallback for bytes not in vocab? Should not happen if vocab covers all bytes 0-255
                    if bpe_token in self.token_to_id:
                        ids.append(self.token_to_id[bpe_token])
                    
        return ids
        
    def encode_iterable(self, iterable) -> Any:
        """Encode an iterable of strings into an iterator of token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string."""
        text_bytes = b"".join([self.vocab[i] for i in ids if i in self.vocab])
        return text_bytes.decode("utf-8", errors="replace")
