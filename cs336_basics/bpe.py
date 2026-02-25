from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel
import pickle as pkl

class BPETokenizerParams(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

def split_data_into_pretokens_and_counts(data: bytes) -> dict[tuple[bytes], int]:
    # split by \n
    data_lines = data.splitlines()

    # split by whitespace
    pretokens = []
    for line in data_lines:
        tokens = line.split(b' ')
        pretokens.extend(tokens)

    pretokens_to_counts = defaultdict(int)

    for token in pretokens:
        split_token = (bytes([b]) for b in token)
        pretokens_to_counts[split_token] += 1

    return pretokens_to_counts


def find_most_frequent_pair(pretokens_to_counts: dict[bytes, int], vocab: dict[int, bytes] | None = None) -> tuple[bytes, bytes]:
    # initialize counts of byte pairs
    counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # count byte pair frequencies
    for byte1, byte2 in zip(data_indices, data_indices[1:]):
        counts[(index1, index2)] += 1

    # find most frequent index pair
    index_pair: tuple[int, int] = (0, 0)
    index_pair_freq = 0

    # for debugging: convert count keys into bytes
    # count_pair_bytes = {}
    # for count_pair, count in counts.items():
    #     id_1, id_2 = count_pair
    #     count_pair_bytes[(vocab[id_1], vocab[id_2])] = count

    for count_pair, count in counts.items():
        if count > index_pair_freq and count_pair > index_pair:
            index_pair_freq = count
            index_pair = count_pair

    return index_pair

def merge(data_indices: list[int], index_pair: tuple[int], target_index: int) -> list[int]:
    new_data_indices = []
    i = 0

    while i < (len(data_indices) - 1):
        if (data_indices[i], data_indices[i+1]) == (index_pair):
            new_data_indices.append(target_index)
            i += 1
        else: 
            new_data_indices.append(data_indices[i])

        i += 1 

    return new_data_indices

def train_bpe(input_path: Path, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:

    with open(input_path, "rb") as f:
        data = f.read()

    # initialize vocab for all the default bytes in UTF-8
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # append special tokens to vocab with keys 256, 257, ...
    for i, token in enumerate(special_tokens):
        token_bytes = token.encode("utf-8")
        vocab[(max(vocab) + i)] = token_bytes

        # remove special token from the data, since it should not affect BPE training
        data.replace(token_bytes, b"")

    pretokens_to_counts = split_data_into_pretokens_and_counts(data)

    while len(vocab) < vocab_size:
        index_pair = find_most_frequent_pair(pretokens_to_counts, vocab)
        id_1, id_2 = index_pair

        target_index = max(vocab) + 1
        vocab[target_index] = vocab[id_1] + vocab[id_2]
        data_indices = merge(pretokens_to_counts, index_pair, target_index)

        bytes_pair = (vocab[id_1], vocab[id_2])
        merges.append(bytes_pair) 

    return BPETokenizerParams(vocab=vocab, merges=merges)

if __name__ == "__main__":
    dataset_path = Path("data/bpe_example.txt")
    bpe_params_save_path = Path("artifacts/bpe_params.json")
    vocab_size = 262
    special_tokens = ["<|endoftext|>"]

    bpe_tokenizer_params = train_bpe(dataset_path, vocab_size, special_tokens)

    bpe_params_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bpe_params_save_path, "wb") as f:
        pkl.dump(bpe_tokenizer_params, f)