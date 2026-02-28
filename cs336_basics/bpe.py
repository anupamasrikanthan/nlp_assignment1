import regex
from typing import Iterable, Iterator
import ast # Useful for safely evaluating serialized string representations if needed
import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Invert vocab for fast token ID lookups during encoding
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        
        # Reconstruct the fast merges dictionary you used in your logic mapping (id1, id2) -> new_id
        # Assuming the merged token ID exists in the vocab
        self.merges_dict = {}
        for (b1, b2) in self.merges:
            combined_bytes = b1 + b2
            if combined_bytes in self.vocab_inverse:
                id1 = self.vocab_inverse[b1]
                id2 = self.vocab_inverse[b2]
                new_id = self.vocab_inverse[combined_bytes]
                self.merges_dict[(id1, id2)] = new_id

        if self.special_tokens:
            # Sort by length descending to match longest special tokens first
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = regex.compile("|".join(map(regex.escape, sorted_specials)))
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Constructs and returns a Tokenizer from a serialized vocabulary and list of merges.
        Note: You will need to adjust the loading logic based on how you chose to serialize your files 
        in the train_bpe function (e.g., JSON, pickle, or raw text).
        """
        # Placeholder loading logic - adjust to match your train_bpe serialization format
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            # Example assuming dict was written as a string
            vocab_str_keys = ast.literal_eval(f.read())
            vocab = {int(k): bytes(v) for k, v in vocab_str_keys.items()}
            
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            # Example assuming list of tuples was written as a string
            merges = ast.literal_eval(f.read())
            
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Using errors="replace" correctly handles malformed bytes with U+FFFD as requested
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yields token IDs from an iterable of strings for memory efficiency."""
        for string in iterable:
            yield from self.encode(string)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if not self.special_pattern:
            return self._encode_chunk(text)
        
        parts = self.special_pattern.split(text)
        specials = self.special_pattern.findall(text)
        
        special_to_id = {v.decode("utf-8"): k for k, v in self.vocab.items() 
                         if v.decode("utf-8", errors="ignore") in self.special_tokens}
        
        result = []
        for i in range(len(parts)):
            if parts[i]:
                result.extend(self._encode_chunk(parts[i]))
            if i < len(specials):
                result.append(special_to_id[specials[i]])
        return result

    def _encode_chunk(self, string: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = regex.findall(PAT, string)
        
        final_ids = []
        for token in tokens:
            ids = [self.vocab_inverse[bytes([b])] for b in token.encode("utf-8")]
            while len(ids) >= 2:
                stats = []
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    if pair in self.merges_dict:
                        # Rank based on order of creation (which matches dictionary insertion order or index)
                        rank = list(self.merges_dict.keys()).index(pair)
                        stats.append((rank, i, pair))
                
                if not stats:
                    break
                
                best_rank, _, best_pair = min(stats)
                new_id = self.merges_dict[best_pair]
                
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

# Pre-compile the pattern at the module level for the worker processes
PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _process_chunk(chunk_args):
    """
    Worker function to process a chunk of text, split by special tokens, 
    and count word frequencies without crossing document boundaries.
    """
    text_chunk, special_tokens = chunk_args
    word_freqs = Counter()
    
    # Strip out all special tokens from the chunk to prevent merging across the text they delimit [cite: 214-215].
    if special_tokens:
        # Split using '|'.join(special_tokens) as the delimiter with careful regex escaping[cite: 217].
        special_pattern = "|".join(map(regex.escape, special_tokens))
        fragments = regex.split(f"({special_pattern})", text_chunk)
    else:
        fragments = [text_chunk]
        
    for frag in fragments:
        if not frag or frag in special_tokens:
            continue
        # Use re.finditer to avoid storing the pre-tokenized words in memory[cite: 164].
        for match in PAT.finditer(frag):
            word_freqs[match.group()] += 1
            
    return word_freqs

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE tokenizer.
    """
    # 1. Initialize Vocabulary [cite: 143-145]
    vocab = {i: bytes([i]) for i in range(256)}
    current_vocab_size = 256
    
    for token in special_tokens:
        vocab[current_vocab_size] = token.encode("utf-8")
        current_vocab_size += 1
        
    num_merges = vocab_size - current_vocab_size
    merges = []

    if num_merges <= 0:
        return vocab, merges

    # 2. Read Text and Chunk for Multiprocessing
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Determine chunk boundaries
    num_procs = max(1, cpu_count() - 1)
    chunks = []
    
    # Chunk the corpus while ensuring boundaries occur at the beginning of a special token [cite: 208-209].
    if special_tokens:
        delim = special_tokens[0]
        docs = text.split(delim)
        
        # Reattach the delimiter to maintain the exact text
        if len(docs) > 1:
            docs = [docs[0]] + [delim + d for d in docs[1:]]
            
        chunk_size = max(1, len(docs) // num_procs)
        for i in range(0, len(docs), chunk_size):
            chunks.append("".join(docs[i:i+chunk_size]))
    else:
        # Fallback if there are no special tokens to chunk safely
        chunk_size = max(1, len(text) // num_procs)
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])

    # 3. Parallelize Pre-tokenization [cite: 206-208]
    chunk_args = [(c, special_tokens) for c in chunks]
    word_freqs = Counter()
    
    with Pool(num_procs) as pool:
        for chunk_freqs in pool.imap_unordered(_process_chunk, chunk_args):
            word_freqs.update(chunk_freqs)

    # `splits` maps a string word -> list of integer token IDs
    splits = {word: list(word.encode("utf-8")) for word in word_freqs}
    
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    # 4. Fast Merge Loop [cite: 219-221]
    for _ in range(num_merges):
        if not pair_freqs:
            break
            
        # Get best pair (Max frequency, then lexicographical tie-break by underlying bytes) [cite: 170]
        best_pair = max(
            pair_freqs.keys(), 
            key=lambda p: (pair_freqs[p], (vocab[p[0]], vocab[p[1]]))
        )
        
        p0, p1 = best_pair
        new_id = current_vocab_size
        current_vocab_size += 1
        
        b1, b2 = vocab[p0], vocab[p1]
        vocab[new_id] = b1 + b2
        merges.append((b1, b2))
        
        words_to_update = list(pair_to_words[best_pair])
        
        del pair_freqs[best_pair]
        del pair_to_words[best_pair]
        
        for word in words_to_update:
            split = splits[word]
            freq = word_freqs[word]
            
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                if pair != best_pair:
                    pair_freqs[pair] -= freq
                    if pair_freqs[pair] <= 0:
                        del pair_freqs[pair]
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(word)
            
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == p0 and split[i+1] == p1:
                    new_split.append(new_id)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
                    
            splits[word] = new_split
            
            for i in range(len(new_split) - 1):
                pair = (new_split[i], new_split[i+1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word)

    return vocab, merges