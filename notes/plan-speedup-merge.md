# Plan: Optimize BPE Training to Pass Speed Tests

## Context

Three speed tests are failing because the BPE implementation is too slow:
- `test__find_most_frequent_pair__speed`: 6 calls of `count_byte_pair_freqs` + `find_most_frequent_pair` must complete in <50ms
- `test__merge__speed`: 6 calls of `merge()` must complete in <70ms
- `test_train_bpe_speed`: full BPE training (500-vocab) on `corpus.en` must complete in <1.5s (reference: 0.38s)

## Root Cause Analysis

### Bottleneck 1: `merge()` — The biggest problem

**Current behavior** (`bpe.py:154-160`):
```python
has_common_bytes = merge_pair[0] in current_pair or merge_pair[1] in current_pair
if not has_common_bytes:
    ...continue
update_byte_pair_freqs(...)  # called for every position where EITHER byte appears
```

`update_byte_pair_freqs` (a Python function call with list allocations and list comparisons) is invoked at every pretoken position where EITHER byte of the merge pair appears — far more than necessary. For merge_pair `(b"e", b" ")`, `e` and space are very common, so this function is called at almost every character position across thousands of pretokens.

### Bottleneck 2: `find_most_frequent_pair()` — Python loop vs C `max()`

The current Python `for` loop (`bpe.py:112-115`) iterates through the entire dict with Python-level dispatch overhead. Python's built-in `max()` runs the same scan in C and is ~3–5× faster.

### Bottleneck 3: `max(vocab)` on every iteration in `train_bpe()`

Minor: `max(vocab)` is called every loop iteration (`bpe.py:234`), costing O(vocab_size) each time.

## Proposed Changes — File: `cs336_basics/bpe.py`

### Change 1: Rewrite `merge()` with simple while-loop + diff-based freq update

Replace the complex character-by-character state machine and the expensive `update_byte_pair_freqs()` calls with:
1. Fast skip: `if a not in pretoken: continue` (avoids processing pretokens that can't contain the pair)
2. A clean `while i < len(pretoken)` loop that just does the merge
3. After the merge, if the pretoken changed: subtract all old adjacent pairs from `byte_pair_freqs`, then add all new adjacent pairs. This is correct for all edge cases including consecutive pairs.

```python
def merge(pretokens_to_counts, merge_pair, byte_pair_freqs):
    a, b = merge_pair
    merged = a + b
    new_pretokens_to_counts = {}

    for pretoken, count in pretokens_to_counts.items():
        if a not in pretoken:
            new_pretokens_to_counts[pretoken] = count
            continue

        new_pretoken = []
        i = 0
        changed = False
        while i < len(pretoken):
            if i + 1 < len(pretoken) and pretoken[i] == a and pretoken[i + 1] == b:
                new_pretoken.append(merged)
                i += 2
                changed = True
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        new_pretoken = tuple(new_pretoken)
        if changed:
            for bp in zip(pretoken, pretoken[1:]):
                byte_pair_freqs[bp] -= count
            for bp in zip(new_pretoken, new_pretoken[1:]):
                byte_pair_freqs[bp] += count

        new_pretokens_to_counts[new_pretoken] = count

    return new_pretokens_to_counts
```

Delete `update_byte_pair_freqs()` entirely.

### Change 2: Replace `find_most_frequent_pair()` with `max()`

```python
def find_most_frequent_pair(byte_pair_freqs):
    return max(byte_pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
```

Same semantics: highest count wins, ties broken by lexicographically larger pair.

### Change 3: Track `next_vocab_idx` in `train_bpe()` instead of calling `max(vocab)` each iteration

```python
next_vocab_idx = max(vocab) + 1
while len(vocab) < vocab_size:
    merge_pair = find_most_frequent_pair(byte_pair_freqs)
    byte_1, byte_2 = merge_pair
    vocab[next_vocab_idx] = byte_1 + byte_2
    next_vocab_idx += 1
    pretokens_to_counts = merge(pretokens_to_counts, merge_pair, byte_pair_freqs)
    merges.append(merge_pair)
```

## Why This Is Correct

- The diff-based freq update (`subtract old pairs, add new pairs`) is equivalent to the incremental update and handles all cases correctly, including consecutive merge pairs like `(a, b, a, b)`.
- `train_bpe` already calls `count_byte_pair_freqs` once before the loop and relies on `merge()` to update `byte_pair_freqs` in-place — this structure is preserved.
- The `a not in pretoken` fast path is correct: if `a` doesn't appear in a pretoken, `(a, b)` certainly doesn't, so no merge or freq update is needed.

## Verification

```bash
# Run correctness tests first
uv run pytest tests/extra/test_train_bpe_extra.py::test_train_bpe_stylized_example -v
uv run pytest tests/test_train_bpe.py::test_train_bpe -v

# Run speed tests
uv run pytest tests/extra/test_train_bpe_extra.py::test__find_most_frequent_pair__speed -v
uv run pytest tests/extra/test_train_bpe_extra.py::test__merge__speed -v
uv run pytest tests/test_train_bpe.py::test_train_bpe_speed -v
```

## Critical File

- `cs336_basics/bpe.py` — functions `merge` (L131), `update_byte_pair_freqs` (L176, delete), `find_most_frequent_pair` (L106), `train_bpe` (L214)
