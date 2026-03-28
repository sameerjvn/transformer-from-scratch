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
    stripped_chunks = re.split(b"|".join(escaped_special_tokens), chunk)

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

    boundary_pairs = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    unique_pretokens_to_counts = defaultdict(int)
    all_pretoken_counts: list[dict[bytes, int]] = []

    if parallel:
        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(dataset_path, special_tokens, boundary_pair) for boundary_pair in boundary_pairs]
            all_pretoken_counts = pool.starmap(count_unique_pretokens_in_chunk, args)
    else:
        for boundary_pair in boundary_pairs:
            unique_pretoken_counts_in_chunk = count_unique_pretokens_in_chunk(
                dataset_path, special_tokens, boundary_pair
            )
            all_pretoken_counts.append(unique_pretoken_counts_in_chunk)

    for pretoken_counts in all_pretoken_counts:
        for key, value in pretoken_counts.items():
            unique_pretokens_to_counts[key] += value

    return unique_pretokens_to_counts


def pretokenize(data: bytes, special_tokens: list[bytes]) -> dict[bytes, int]:

    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    stripped_data_items = re.split(b"|".join(escaped_special_tokens), data)

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


def parallel_pretokenize_and_split(
    dataset_path: Path, special_tokens: list[bytes], parallel: bool = True
) -> dict[tuple[bytes], int]:
    pretokens_to_counts = parallel_pretokenize(dataset_path, special_tokens, parallel)
    split_pretokens_to_counts = split_pretokens(pretokens_to_counts)
    return split_pretokens_to_counts


def find_most_frequent_pair(byte_pair_freqs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:

    # find most frequent index pair
    most_freq_pair: tuple[bytes, bytes] = tuple()
    most_freq_count = 0

    for byte_pair, count in byte_pair_freqs.items():
        if count > most_freq_count or (count == most_freq_count and byte_pair > most_freq_pair):
            most_freq_count = count
            most_freq_pair = byte_pair

    return most_freq_pair


def count_byte_pair_freqs(pretokens_to_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    # initialize counts of byte pairs
    byte_pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # count byte pair frequencies
    for pretoken, count in pretokens_to_counts.items():
        for byte_pair in zip(pretoken, pretoken[1:]):
            byte_pair_freqs[byte_pair] += count
    return byte_pair_freqs


def merge(
    pretokens_to_counts: dict[tuple[bytes], int],
    merge_pair: tuple[bytes, bytes],
    byte_pair_freqs: dict[tuple[bytes, bytes], int],
) -> dict[tuple[bytes], int]:
    new_pretokens_to_counts = {}

    for pretoken, counts in pretokens_to_counts.items():
        new_pretoken = list()  # tuples are immutable, so increment a list and then convert a tuple

        prev_pair_merged = False
        for i in range(len(pretoken)):
            current_byte = pretoken[i]
            current_pair = None

            is_last_byte = i == (len(pretoken) - 1)
            if is_last_byte:
                if not prev_pair_merged:
                    new_pretoken.append(current_byte)
                break

            current_pair = (pretoken[i], pretoken[i + 1])

            has_common_bytes = merge_pair[0] in current_pair or merge_pair[1] in current_pair
            if not has_common_bytes:
                continue

            update_byte_pair_freqs(merge_pair, byte_pair_freqs, pretoken, counts, i, current_pair)

            if current_pair == merge_pair:
                new_pretoken.append(b"".join(list(current_pair)))
                prev_pair_merged = True

        new_pretokens_to_counts[tuple(new_pretoken)] = counts

    return new_pretokens_to_counts


def update_byte_pair_freqs(merge_pair, byte_pair_freqs, pretoken, counts, i, current_pair):
    left_pair, right_pair = None, None
    if i > 0:
        left_pair = [pretoken[i - 1], pretoken[i]]
    if i < (len(pretoken) - 2):
        right_pair = [pretoken[i + 1], pretoken[i + 2]]

    decreasing_pair, increasing_pair = None, None

    # freq update logic
    right_increasing_byte, left_increasing_byte = None, None
    if left_pair == merge_pair:
        decreasing_pair = current_pair
        left_increasing_byte = b"".join(left_pair)
    elif left_pair is None:
        left_increasing_byte = current_pair[0]

    if right_pair == merge_pair:
        decreasing_pair = current_pair
        right_increasing_byte = b"".join(right_pair)
    elif right_pair is None:
        right_increasing_byte = current_pair[1]

    increasing_pair = [left_increasing_byte, right_increasing_byte]

    if decreasing_pair is not None:
        byte_pair_freqs[tuple(decreasing_pair)] -= counts

    if increasing_pair is not None:
        byte_pair_freqs[tuple(increasing_pair)] += counts


def train_bpe(
    input_path: Path, vocab_size: int, special_tokens: list[str], parallel: bool = False
) -> BPETokenizerParams:

    # initialize vocab for all the default bytes in UTF-8
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # append special tokens to vocab with keys 256, 257, ...
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    for i, token_bytes in enumerate(special_tokens_bytes):
        vocab[(max(vocab) + i + 1)] = token_bytes

    pretokens_to_counts = parallel_pretokenize_and_split(input_path, special_tokens_bytes, parallel)
    byte_pair_freqs = count_byte_pair_freqs(pretokens_to_counts)

    while len(vocab) < vocab_size:
        merge_pair = find_most_frequent_pair(byte_pair_freqs)
        byte_1, byte_2 = merge_pair

        target_index = max(vocab) + 1
        vocab[target_index] = byte_1 + byte_2
        pretokens_to_counts = merge(pretokens_to_counts, merge_pair, byte_pair_freqs)

        merges.append(merge_pair)

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    dataset_path = Path("data/TinyStories-valid.txt")
    vocab_size = 262
    special_tokens = ["<|endoftext|>"]

    bpe_tokenizer_params = train_bpe(dataset_path, vocab_size, special_tokens)
