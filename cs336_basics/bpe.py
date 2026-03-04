from collections import defaultdict
import multiprocessing
from pathlib import Path
from pydantic import BaseModel
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

class BPETokenizerParams(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

PAT = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def count_unique_pretokens_in_chunk(dataset_path, special_tokens, boundary_pair) -> dict[bytes, int]:
    start, end = boundary_pair
    with open(dataset_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    stripped_chunks = re.split(b'|'.join(escaped_special_tokens), chunk)

    unique_pretokens_to_counts_in_chunk = defaultdict(int)

    for stripped_chunk in stripped_chunks:
        pretokens = PAT.finditer(stripped_chunk)
        for token in pretokens:
            token_bytes = token.group()
            unique_pretokens_to_counts_in_chunk[token_bytes] += 1

    return unique_pretokens_to_counts_in_chunk

def parallel_pretokenize(dataset_path: Path, special_tokens: list[bytes], parallel: bool = True) -> dict[bytes, int]:
    num_processes = multiprocessing.cpu_count()

    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    boundary_pairs = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    unique_pretokens_to_counts = defaultdict(int)
    all_pretoken_counts: list[dict[bytes, int]] = []

    if parallel:
        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(dataset_path, special_tokens, boundary_pair) for boundary_pair in boundary_pairs]
            all_pretoken_counts = pool.starmap(count_unique_pretokens_in_chunk, args)
    else:
        for boundary_pair in boundary_pairs:
            unique_pretoken_counts_in_chunk = count_unique_pretokens_in_chunk(dataset_path, special_tokens, boundary_pair)
            all_pretoken_counts.append(unique_pretoken_counts_in_chunk)

    for pretoken_counts in all_pretoken_counts:
        for key, value in pretoken_counts.items():
            unique_pretokens_to_counts[key] += value
        
    return unique_pretokens_to_counts

def pretokenize(data: bytes, special_tokens: list[bytes]) -> dict[bytes, int]:

    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    stripped_data_items = re.split(b'|'.join(escaped_special_tokens), data)

    unique_pretokens_to_counts = defaultdict(int)

    for stripped_data in stripped_data_items:
        pretokens = PAT.finditer(stripped_data)
        for token in pretokens:
            token_bytes = token.group()
            unique_pretokens_to_counts[token_bytes] += 1

    return unique_pretokens_to_counts

def split_pretokens(pretokens_to_counts: dict[bytes, int]) -> dict[tuple[bytes], int]:
    split_pretokens_to_counts = defaultdict(int)
    for token_bytes, counts in pretokens_to_counts.items():
        split_token = tuple(bytes([b]) for b in token_bytes)
        split_pretokens_to_counts[split_token] = counts

    return split_pretokens_to_counts

def pretokenize_and_split(data: bytes, special_tokens: list[bytes]) -> dict[tuple[bytes], int]:
    pretokens_to_counts = pretokenize(data, special_tokens)
    split_pretokens_to_counts = split_pretokens(pretokens_to_counts)
    return split_pretokens_to_counts

def parallel_pretokenize_and_split(dataset_path: Path, special_tokens: list[bytes], parallel: bool = True) -> dict[tuple[bytes], int]:
    pretokens_to_counts = parallel_pretokenize(dataset_path, special_tokens, parallel)
    split_pretokens_to_counts = split_pretokens(pretokens_to_counts)
    return split_pretokens_to_counts

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
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    for i, token_bytes in enumerate(special_tokens_bytes):
        vocab[(max(vocab) + i + 1)] = token_bytes

        # remove special token from the data, since it should not affect BPE training
        data = data.replace(token_bytes, b"")

    pretokens_to_counts = pretokenize_and_split(data, special_tokens_bytes)

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