import regex
from typing import Iterable, Iterator
import ast
import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.merges_dict = {}
        for (b1, b2) in self.merges:
            combined_bytes = b1 + b2
            if combined_bytes in self.vocab_inverse:
                id1 = self.vocab_inverse[b1]
                id2 = self.vocab_inverse[b2]
                new_id = self.vocab_inverse[combined_bytes]
                self.merges_dict[(id1, id2)] = new_id
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = regex.compile("|".join(map(regex.escape, sorted_special_tokens)))
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None): 
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_keys = ast.literal_eval(f.read())
            vocab = {int(key): bytes(value) for key, value in vocab_keys.items()}    
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = ast.literal_eval(f.read())   
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def encode(self, text: str) -> list[int]:
        if not self.special_pattern:
            return self.encode_chunk(text)
        parts = self.special_pattern.split(text)
        specials = self.special_pattern.findall(text)
        special_to_id = {value.decode("utf-8"): key for key, value in self.vocab.items() if value.decode("utf-8", errors="ignore") in self.special_tokens}
        result = []
        for i in range(len(parts)):
            if parts[i]:
                result.extend(self.encode_chunk(parts[i]))
            if i < len(specials):
                result.append(special_to_id[specials[i]])
        return result

    def encode_chunk(self, string: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  #as given in assignment
        tokens = regex.findall(PAT, string)
        final_ids = []
        for token in tokens:
            ids = [self.vocab_inverse[bytes([b])] for b in token.encode("utf-8")]
            while len(ids)>= 2:
                temp = []
                for i in range(len(ids)-1):
                    pair = (ids[i], ids[i+1])
                    if pair in self.merges_dict:
                        rank_char = list(self.merges_dict.keys()).index(pair)
                        temp.append((rank_char,i,pair))
                if not temp:
                    break
                best_rank, _, best_pair = min(temp)
                new_id = self.merges_dict[best_pair]
                new_ids = []
                i = 0
                while i<len(ids):
                    if i<len(ids)-1 and (ids[i],ids[i+1]) == best_pair:
                        new_ids.append(new_id)
                        i+= 2
                    else:
                        new_ids.append(ids[i])
                        i+= 1
                ids = new_ids
            final_ids.extend(ids)
        return final_ids

PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")   #as given in assignment

def _process_chunk(chunk_args):
    text_chunk,special_tokens = chunk_args
    word_freqs = Counter()
    if special_tokens:
        special_pattern = "|".join(map(regex.escape,special_tokens))
        fragments = regex.split(f"({special_pattern})", text_chunk)
    else:
        fragments = [text_chunk]
    for frag in fragments:
        if not frag or frag in special_tokens:
            continue
        for match in PAT.finditer(frag):
            word_freqs[match.group()]+= 1    
    return word_freqs

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    current_vocab_size=256
    for token in special_tokens:
        vocab[current_vocab_size] = token.encode("utf-8")
        current_vocab_size+=1
    num_merges = vocab_size-current_vocab_size
    merges = []
    if num_merges<=0:
        return vocab, merges
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    num_procs = max(1,cpu_count()-1)
    chunks = []
    if special_tokens:
        delim = special_tokens[0]
        docs = text.split(delim)
        if len(docs)>1:
            docs = [docs[0]]+[delim+d for d in docs[1:]] 
        chunk_size = max(1,len(docs)//num_procs)
        for i in range(0,len(docs), chunk_size):
            chunks.append("".join(docs[i:i+chunk_size]))
    else:
        chunk_size = max(1,len(text)//num_procs)
        for i in range(0,len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    chunk_args = [(c,special_tokens) for c in chunks]
    word_freqs = Counter()
    with Pool(num_procs) as pool:
        for chunk_freqs in pool.imap_unordered(_process_chunk, chunk_args):
            word_freqs.update(chunk_freqs)
    splits = {word: list(word.encode("utf-8")) for word in word_freqs}
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split)-1):
            pair = (split[i],split[i+1])
            pair_freqs[pair]+= freq
            pair_to_words[pair].add(word)
    for _ in range(num_merges):
        if not pair_freqs:
            break    
        best_pair = max(pair_freqs.keys(),key=lambda p: (pair_freqs[p],(vocab[p[0]],vocab[p[1]])))
        pair0,pair1 = best_pair
        new_id = current_vocab_size
        current_vocab_size+=1
        b1,b2 = vocab[pair0],vocab[pair1]
        vocab[new_id] = b1+b2
        merges.append((b1,b2))
        words_to_update = list(pair_to_words[best_pair])
        del pair_freqs[best_pair]
        del pair_to_words[best_pair]
        for word in words_to_update:
            split = splits[word]
            freq = word_freqs[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                if pair!=best_pair:
                    pair_freqs[pair]-=freq
                    if pair_freqs[pair]<= 0:
                        del pair_freqs[pair]
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(word)
            temp_split = []
            i=0
            while i<len(split):
                if i<len(split)-1 and split[i] == pair0 and split[i+1] == pair1:
                    temp_split.append(new_id)
                    i+= 2
                else:
                    temp_split.append(split[i])
                    i+= 1     
            splits[word] = temp_split
            for i in range(len(temp_split)-1):
                pair = (temp_split[i], temp_split[i+1])
                pair_freqs[pair]+= freq
                pair_to_words[pair].add(word)
    return vocab, merges