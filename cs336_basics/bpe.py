from abc import ABC
from dataclasses import dataclass
from collections import defaultdict

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        # String -> list of bytes -> list of indices
        indices = list(map(int, string.encode("utf-8")))  
        
        # Merge indices according to merges. Each merge replaces a pair of indices with a new index.
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  
            indices = merge(indices, pair, new_index)
        return indices
    
    def decode(self, indices: list[int]) -> str:
        # List of indices -> list of bytes -> string
        bytes_list = list(map(self.params.vocab.get, indices))  
        string = b"".join(bytes_list).decode("utf-8") 
        return string

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  
    i = 0 
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def bpe_tokenizer():
    """
    Byte Pair Encoding (BPE)
    The BPE algorithm was introduced by Philip Gage in 1994 for data compression.
    It was adapted to NLP for neural machine translation.  [Sennrich+ 2015]
    (Previously, papers had been using word-based tokenization.)
    BPE was then used by GPT-2.  [Radford+ 2019]
    Basic idea: train the tokenizer on raw text to automatically determine the vocabulary.
    Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.
    The GPT-2 paper used word-based tokenization to break up the text into inital segments and run the original BPE algorithm on each segment.

    Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.
    """

    # Training the tokenizer
    string = "the cat in the hat"
    params = train_bpe(string, num_merges=3)

    # Using the tokenizer
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  
    indices = tokenizer.encode(string)  
    reconstructed_string = tokenizer.decode(indices)  
    assert string == reconstructed_string

    """
    In Assignment 1, you will go beyond this in the following ways:  
    encode() currently loops over all merges. Only loop over merges that matter.
    Detect and preserve special tokens (e.g., <|endoftext|>).
    Use pre-tokenization (e.g., the GPT-2 tokenizer regex).
    Try to make the implementation as fast as possible.
    """

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  
    merges: dict[tuple[int, int], int] = {}  
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # Note the difference between bytes(x) and bytes([x])!
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  
            counts[(index1, index2)] += 1 
        
        # Find the most common pair.
        pair = max(counts, key=counts.get) 
        index1, index2 = pair
        
        # Merge that pair.
        new_index = 256 + i 
        merges[pair] = new_index  
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)
    
    return BPETokenizerParams(vocab=vocab, merges=merges)


bpe_tokenizer()