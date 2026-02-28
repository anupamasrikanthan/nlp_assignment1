from abc import ABC
from dataclasses import dataclass
from collections import Counter, defaultdict
from collections.abc import Iterable
import regex

class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]

class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams, special_tokens: list[str] | None = None):
        self.params = params
        self.special_tokens = special_tokens or []
        # Create inverse vocab for efficient ID lookup
        self.vocab_inverse = {v: k for k, v in self.params.vocab.items()}
        
        if self.special_tokens:
            # Sort by length descending to match longest possible special token first
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = regex.compile("|".join(map(regex.escape, sorted_specials)))
        else:
            self.special_pattern = None

    def decode(self, indices: list[int]) -> str:
        # Concatenate bytes first to handle multi-byte UTF-8 sequences
        raw_bytes = b"".join(self.params.vocab[idx] for idx in indices)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        all_ids = []
        for string in iterable:
            all_ids.extend(self.encode(string))
        return all_ids

    def encode(self, string: str) -> list[int]:
        if not self.special_pattern:
            return self._encode_chunk(string)
        
        # Split string by special tokens while preserving them
        parts = self.special_pattern.split(string)
        specials = self.special_pattern.findall(string)
        
        # Map special token strings to their corresponding IDs
        special_to_id = {v.decode("utf-8"): k for k, v in self.params.vocab.items() 
                         if v.decode("utf-8", errors="ignore") in self.special_tokens}
        
        result = []
        for i in range(len(parts)):
            if parts[i]:
                result.extend(self._encode_chunk(parts[i]))
            if i < len(specials):
                result.append(special_to_id[specials[i]])
        return result

    def _encode_chunk(self, string: str) -> list[int]:
        # GPT-2 pre-tokenization regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = regex.findall(PAT, string)
        
        final_ids = []
        for token in tokens:
            ids = [self.vocab_inverse[bytes([b])] for b in token.encode("utf-8")]
            while len(ids) >= 2:
                stats = []
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    if pair in self.params.merges:
                        # Rank based on order in the merge dictionary
                        rank = list(self.params.merges.keys()).index(pair)
                        stats.append((rank, i, pair))
                
                if not stats:
                    break
                
                # Prioritize merge with the lowest rank
                best_rank, _, best_pair = min(stats)
                new_id = self.params.merges[best_pair]
                
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            final_ids.extend(ids)
        return final_ids

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 1. Initialize vocab with individual bytes and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    
    current_vocab_size = 256 + len(special_tokens)
    num_merges = vocab_size - current_vocab_size
    merges = []

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Treat special tokens as atomic and isolate regex segments
    special_pattern = "|".join(map(regex.escape, special_tokens)) if special_tokens else ""
    fragments = regex.split(f"({special_pattern})", text) if special_pattern else [text]
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    word_counts = Counter()
    for frag in fragments:
        if frag and frag not in (special_tokens or []):
            words = regex.findall(PAT, frag)
            for word in words:
                word_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    # 3. Merging Loop (Optimized)
    # Pre-calculate pair frequencies once
    pair_freqs = defaultdict(int)
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair_freqs[word_tuple[i:i+2]] += count

    for _ in range(num_merges):
        # Filter out stale pairs that dropped to 0 or below
        valid_pairs = {k: v for k, v in pair_freqs.items() if v > 0}
        if not valid_pairs:
            break

        best_pair = max(valid_pairs.keys(), key=lambda p: (valid_pairs[p], p))
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[current_vocab_size] = new_token

        new_word_counts = Counter()
        for word_tuple, count in word_counts.items():
            # Only process words that actually contain the pair
            if best_pair[0] in word_tuple and best_pair[1] in word_tuple:
                pair_in_word = False
                for i in range(len(word_tuple) - 1):
                    if word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                        pair_in_word = True
                        break
                
                if pair_in_word:
                    # Subtract old pair frequencies
                    for i in range(len(word_tuple) - 1):
                        pair_freqs[word_tuple[i:i+2]] -= count
                        
                    # Create the new merged word
                    new_word = []
                    i = 0
                    while i < len(word_tuple):
                        if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                            new_word.append(new_token)
                            i += 2
                        else:
                            new_word.append(word_tuple[i])
                            i += 1
                    
                    new_word_tuple = tuple(new_word)
                    new_word_counts[new_word_tuple] = count
                    
                    # Add new pair frequencies
                    for i in range(len(new_word_tuple) - 1):
                        pair_freqs[new_word_tuple[i:i+2]] += count
                    continue
                    
            # If the pair wasn't in the word, just carry it over unchanged
            new_word_counts[word_tuple] = count

        word_counts = new_word_counts
        current_vocab_size += 1

    return vocab, merges