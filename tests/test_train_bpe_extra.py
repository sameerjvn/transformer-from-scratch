from .adapters import run_train_bpe
from .common import FIXTURES_PATH

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