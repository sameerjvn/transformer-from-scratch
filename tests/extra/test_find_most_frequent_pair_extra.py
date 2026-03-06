import pytest
import time
import random
from collections import Counter

from cs336_basics.bpe import find_most_frequent_pair


# --- Helpers ---

def generate_pretokens_to_counts(num_pretokens: int, tuple_len_range=(2, 8), seed=42) -> dict[tuple[bytes], int]:
    """Generate a realistic pretokens_to_counts dict of a given size."""
    rng = random.Random(seed)
    vocab = [bytes([i]) for i in range(256)]  # single-byte tokens
    result = {}
    for _ in range(num_pretokens):
        length = rng.randint(*tuple_len_range)
        key = tuple(rng.choice(vocab) for _ in range(length))
        result[key] = rng.randint(1, 10_000)
    return result


# --- Raw Throughput Tests ---

class TestRawThroughput:

    def test_medium_input_speed(self):
        """10k pretokens should complete in under 100ms."""
        data = generate_pretokens_to_counts(10_000)
        start = time.perf_counter()
        result = find_most_frequent_pair(data)
        elapsed = time.perf_counter() - start

        assert result is not None
        assert elapsed < 0.1, f"Too slow on 10k input: {elapsed:.3f}s"

    def test_large_input_speed(self):
        """100k pretokens should complete in under 500ms."""
        data = generate_pretokens_to_counts(100_000)
        start = time.perf_counter()
        result = find_most_frequent_pair(data)
        elapsed = time.perf_counter() - start

        assert result is not None
        assert elapsed < 0.5, f"Too slow on 100k input: {elapsed:.3f}s"

    def test_very_large_input_speed(self):
        """500k pretokens should complete in under 2s."""
        data = generate_pretokens_to_counts(500_000)
        start = time.perf_counter()
        result = find_most_frequent_pair(data)
        elapsed = time.perf_counter() - start

        assert result is not None
        assert elapsed < 2.0, f"Too slow on 500k input: {elapsed:.3f}s"


# --- Scaling Behavior Tests ---

class TestScalingBehavior:

#    SIZES = [10_000, 50_000, 100_000, 500_000]
    SIZES = [10, 100, 1000]

    def _measure(self, size: int, runs: int = 3) -> float:
        """Average elapsed time over multiple runs."""
        data = generate_pretokens_to_counts(size)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            find_most_frequent_pair(data)
            times.append(time.perf_counter() - start)
        return sum(times) / len(times)

    def test_scaling_is_roughly_linear(self):
        """
        Doubling input size should not more than triple the runtime.
        This catches accidentally O(n^2) implementations.
        """
        t_10k  = self._measure(10_000)
        t_100k = self._measure(100_000)  # 10x larger

        ratio = t_100k / t_10k
        assert ratio < 30, (
            f"Scaling looks worse than linear: 10x more data took {ratio:.1f}x longer "
            f"({t_10k*1000:.1f}ms -> {t_100k*1000:.1f}ms)"
        )

    def test_no_superlinear_blowup_at_500k(self):
        """Going from 100k to 500k (5x) should not take more than 15x longer."""
        t_100k = self._measure(100_000)
        t_500k = self._measure(500_000)

        ratio = t_500k / t_100k
        assert ratio < 15, (
            f"Unexpected blowup at 500k: {ratio:.1f}x slower than 100k "
            f"({t_100k*1000:.1f}ms -> {t_500k*1000:.1f}ms)"
        )

    def test_print_scaling_table(self, capsys):
        """Not a pass/fail test — prints a timing table for manual inspection."""
        print("\n--- Scaling Table ---")
        print(f"{'Size':>10}  {'Avg Time (ms)':>15}")
        print("-" * 30)
        for size in self.SIZES:
            avg = self._measure(size)
            print(f"{size:>10,}  {avg * 1000:>15.2f}")