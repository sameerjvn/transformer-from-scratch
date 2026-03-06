from pathlib import Path
import time
import regex as re
from cs336_basics.bpe import find_most_frequent_pair, parallel_pretokenize, pretokenize
from ..adapters import run_train_bpe
from ..common import FIXTURES_PATH


def test_train_bpe_stylized_example():
    """
    Ensure that the stylized example from the assignment gives the expected vocabulary at the end of 6 merges
    """
    num_merges = 6
    special_tokens = ["<|endoftext|>"]
    input_path = FIXTURES_PATH / "bpe_example.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=256 + num_merges + len(special_tokens),
        special_tokens=special_tokens,
    )

    assert merges == [(b's', b't'),
                      (b'e', b'st'),
                      (b'o', b'w'),
                      (b'l', b'ow'),
                      (b'w', b'est'),
                      (b'n', b'e'),
                      ]
    
    all_byte_values = {bytes([i]) for i in range(256)}
    special_token_values = {"<|endoftext|>".encode("utf-8")}
    merged_values = {b'st', b'est', b'ow', b'low', b'west', b'ne'}
    expected_vocab = all_byte_values.union(special_token_values).union(merged_values)

    assert set(vocab.values()) == expected_vocab

def test_parallel_pretokenize_correctness():
    dataset_path = FIXTURES_PATH / "tinystories_sample.txt"
    special_tokens = [b"<|endoftext|>"]
    
    with open(dataset_path, "rb") as f:
        data = f.read()

    pretokens_to_counts = pretokenize(data, special_tokens)
    serial_pretokens_to_counts = parallel_pretokenize(dataset_path, special_tokens, parallel=False)

    assert len(pretokens_to_counts) == len(serial_pretokens_to_counts)
    for k, v in serial_pretokens_to_counts.items():
        assert pretokens_to_counts[k] == v

    parallel_pretokens_to_counts = parallel_pretokenize(dataset_path, special_tokens, parallel=True)
    assert len(pretokens_to_counts) == len(parallel_pretokens_to_counts)
    for k, v in parallel_pretokens_to_counts.items():
        assert pretokens_to_counts[k] == v

def test_pretokenize_speed():
    input_path = FIXTURES_PATH / "corpus.en"

    with open(input_path, "rb") as f:
        data = f.read()

    start_time = time.time()
    _ = pretokenize(data, special_tokens=[])
    end_time = time.time()

    time_secs = end_time - start_time
    assert time_secs < 1
    

def test_parallel_pretokenize_speed():
    input_path = FIXTURES_PATH / "corpus.en"

    start_time = time.time()
    pretokens_to_counts = parallel_pretokenize(input_path, special_tokens=[b"<|endoftext|>"])
    end_time = time.time()

    time_secs = end_time - start_time
    assert time_secs < 1
