import time
import regex as re
from cs336_basics.bpe import pretokenize
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

def test_pretokenize_speed():
    input_path = FIXTURES_PATH / "corpus.en"

    with open(input_path, "rb") as f:
        data = f.read()

    start_time = time.time()

    for _ in range(500-256):
        pretokens_to_counts = pretokenize(data)

    end_time = time.time()
    time_secs = end_time - start_time

    assert time_secs < 1

def test_regex_works_on_bytes():
    data = b"some text that i'll pre-tokenize"
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    expected = [b'some', b' text', b' that', b' i', b"'ll", b' pre', b'-', b'tokenize']

    matches = re.findall(PAT, data)

    assert matches == expected