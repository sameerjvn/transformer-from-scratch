from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel

class BPETokenizerParams(BaseModel):
    vocabulary: dict[int, bytes]
    merges: dict[(int, int), int]

def split_data_into_pretokens(data: bytes, ) -> list[bytes]:
    # split by \n
    data_lines = data.splitlines()

    # split by whitespace
    pretokens = []
    for line in data_lines:
        tokens = line.split(b' ')
        pretokens.extend(tokens)

    # remove <|endoftext|>. Why? Just for the example
    pretokens = [pretoken for pretoken in pretokens if pretoken != "<|endoftext|>"]
    return pretokens

def merge(data_indices: list[int], index_pair: tuple[int], target_index: int) -> list[int]:
    new_data_indices = []
    i = 0

    for j in range(len(data_indices) - 1):
        if (data_indices[j], data_indices[j+1]) == (index_pair):
            new_data_indices[i] = target_index
        else: 
            new_data_indices[i] = data_indices[j]
        
        i += 1

    return new_data_indices


def train_bpe(path: Path, num_merges: int):

    vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # add a token ID for <|endoftext|>
    vocabulary[(max(vocabulary) + 1)] = b"<|endoftext|>"

    merges: list[tuple[int, int], int] = []

    with open(path, "rb") as f:
        data = f.read()
    # remove special token from the data, since it is included in the vocabulary
    data.replace(b"<|endoftext|>", b"")

    # convert utf-8 bytestring data to int tokens
    data_indices = list(map(int, data))

    for _ in range(num_merges):
        counts = defaultdict(int)
        # count pair frequencies
        for index1, index2 in zip(data_indices, data_indices[1:]):
            counts[(index1, index2)] += 1

        # find most frequent index pair
        index_pair = max(counts, key=counts.get)
        id_1, id_2 = index_pair

        # add the pair to vocabulary and merge in the dataset
        target_index = max(vocabulary + 1)
        vocabulary[target_index] = vocabulary[id_1] + vocabulary[id_2]
        data_indices = merge(data_indices, index_pair, target_index)
        merges.append(index_pair) = target_index

    return BPETokenizerParams(vocabulary=vocabulary, merges=merges)

if __name__ == "__main__":
    dataset_path = Path("data/bpe_example.txt")
    train_bpe(dataset_path)