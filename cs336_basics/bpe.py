from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel
import regex as re

class BPETokenizerParams(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(data: bytes) -> dict[tuple[bytes], int]:

    pretokens = re.finditer(PAT, data)

    pretokens_to_counts = defaultdict(int)

    for token in pretokens:
        split_token = tuple(bytes([b]) for b in token)
        pretokens_to_counts[split_token] += 1

    return pretokens_to_counts


def find_most_frequent_pair(pretokens_to_counts: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
    # initialize counts of byte pairs
    counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    byte_pairs = set() # 

    # count byte pair frequencies
    for pretoken in pretokens_to_counts.keys():
        for byte_pair in zip(pretoken, pretoken[1:]):
            byte_pairs.add(byte_pair)
    
    for byte_pair in byte_pairs:
        for pretoken, count in pretokens_to_counts.items():
            for bp in zip(pretoken, pretoken[1:]):
                if byte_pair == bp:
                    counts[byte_pair] += count

            
    # find most frequent index pair
    bytes_pair: tuple[bytes, bytes] = tuple()
    bytes_pair_freq = 0

    for count_pair, count in counts.items():
        if count > bytes_pair_freq or (count == bytes_pair_freq and count_pair > bytes_pair):
            bytes_pair_freq = count
            bytes_pair = count_pair

    return bytes_pair

def merge(pretokens_to_counts: dict[tuple[bytes], int], bytes_pair: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    new_pretokens_to_counts = {}

    for pretoken, counts in pretokens_to_counts.items():
        new_pretoken = list() # tuples are immutable, so increment a list and then convert a tuple

        i = 0
        while i < (len(pretoken) - 1):
            bp = [pretoken[i], pretoken[i+1]]
            if tuple(bp) == bytes_pair:
                new_pretoken.append(pretoken[i] + pretoken[i+1])
                i += 1
            elif (i == len(pretoken) - 2):
                new_pretoken.extend(bp)
                i += 1
            else:
                new_pretoken.append(pretoken[i])
            i += 1
        
        new_pretokens_to_counts[tuple(new_pretoken)] = counts

    return new_pretokens_to_counts

def train_bpe(input_path: Path, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:

    with open(input_path, "rb") as f:
        data = f.read()

    # initialize vocab for all the default bytes in UTF-8
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # append special tokens to vocab with keys 256, 257, ...
    for i, token in enumerate(special_tokens):
        token_bytes = token.encode("utf-8")
        vocab[(max(vocab) + i + 1)] = token_bytes

        # remove special token from the data, since it should not affect BPE training
        data = data.replace(token_bytes, b"")

    pretokens_to_counts = pretokenize(data)

    while len(vocab) < vocab_size:
        bytes_pair = find_most_frequent_pair(pretokens_to_counts)
        byte_1, byte_2 = bytes_pair

        target_index = max(vocab) + 1
        vocab[target_index] = byte_1 + byte_2
        pretokens_to_counts = merge(pretokens_to_counts, bytes_pair)

        merges.append(bytes_pair) 

    return BPETokenizerParams(vocab=vocab, merges=merges)

if __name__ == "__main__":
    dataset_path = Path("data/TinyStories-valid.txt")
    vocab_size = 262
    special_tokens = ["<|endoftext|>"]

    bpe_tokenizer_params = train_bpe(dataset_path, vocab_size, special_tokens)