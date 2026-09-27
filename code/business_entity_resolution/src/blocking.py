import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

MAX_BLOCK_SIZE = 40
RARE_WORD_DF = 25
WORKERS = max(1, (os.cpu_count() or 2) - 1)
JOIN_THREADS = 8
MIN_SCORE_RATIO = 0.4
CANDIDATE_COLUMNS = ["target", "source1", "key_score", "shared_keys", "key_rank",
                     "best_key_score", "source1_hits"]

SOURCE1_WIDTHS = dict(name=5, address=8, numbers=3, words_per_number=99,
                      cross_name=3, cross_address=6, name_number=3)
TARGET_WIDTHS = dict(name=4, address=4, numbers=2, words_per_number=3,
                     cross_name=2, cross_address=2, name_number=2)

_worker_state = {}


def word_pairs(words):
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = words[i], words[j]
            pairs.append(a + "|" + b if a < b else b + "|" + a)
    return pairs


def split_joined_word(word, vocabulary, min_piece=3):
    n = len(word)
    best: list[list[str] | None] = [None] * (n + 1)
    best[0] = []
    for start in range(n):
        prev = best[start]
        if prev is None:
            continue
        for end in range(start + min_piece, n + 1):
            piece = word[start:end]
            if piece in vocabulary:
                current = best[end]
                candidate = prev + [piece]
                if current is None or len(current) > len(prev) + 1:
                    best[end] = candidate
    result = best[n]
    return result if result and len(result) >= 2 else []


def rare_name_words(core, name_df):
    words = set()
    for word in core.split():
        if word in name_df:
            words.add(word)
        elif len(word) >= 7:
            words.update(split_joined_word(word, name_df))
    return sorted(words, key=lambda w: (name_df[w], w))


def build_keys(core, addr, nums, name_df, addr_df, widths):
    name_words = rare_name_words(core, name_df)
    addr_words = sorted({w for w in addr.split() if w in addr_df and not w.isdigit()},
                        key=lambda w: (addr_df[w], w))
    numbers = nums.split()
    keys = []
    if len(name_words) == 1:
        keys.append("M" + name_words[0])
    keys += ["N" + p for p in word_pairs(name_words[:widths["name"]])]
    keys += ["U" + w for w in name_words if name_df[w] <= RARE_WORD_DF]
    keys += ["A" + p for p in word_pairs(addr_words[:widths["address"]])]
    for number in numbers[:widths["numbers"]]:
        keys += ["H" + number + "|" + w for w in addr_words[:widths["words_per_number"]]]
        keys += ["Y" + w + "|" + number for w in name_words[:widths["name_number"]]]
    keys += ["X" + a + "|" + b for a in name_words[:widths["cross_name"]]
             for b in addr_words[:widths["cross_address"]]]
    return keys


def _init_worker(name_df, addr_df):
    _worker_state["name_df"], _worker_state["addr_df"] = name_df, addr_df


def _keys_for_chunk(job):
    cores, addrs, nums, offset, is_source1 = job
    name_df, addr_df = _worker_state["name_df"], _worker_state["addr_df"]
    widths = SOURCE1_WIDTHS if is_source1 else TARGET_WIDTHS
    all_keys, owners = [], []
    for i, (core, addr, num) in enumerate(zip(cores, addrs, nums)):
        keys = build_keys(core, addr, num, name_df, addr_df, widths)
        all_keys += keys
        owners += [offset + i] * len(keys)
    hashes = pd.util.hash_array(np.array(all_keys, dtype=object)) if all_keys else np.zeros(0, np.uint64)
    return hashes, np.array(owners, dtype=np.int64)


def compute_keys(pool, cores, addrs, nums, is_source1, chunk_size=40000):
    jobs = [(cores[i:i + chunk_size], addrs[i:i + chunk_size], nums[i:i + chunk_size], i, is_source1)
            for i in range(0, len(cores), chunk_size)]
    results = pool.map(_keys_for_chunk, jobs)
    if not results:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def document_frequency(strings):
    counts = Counter()
    for s in strings:
        counts.update(set(s.split()))
    return dict(counts)


def join_top_candidates(block_keys, block_starts, block_sizes, block_weights, source1_by_key,
                        target_keys, target_owner, top_k):
    if len(block_keys) == 0 or len(target_keys) == 0:
        return None
    pos = np.minimum(np.searchsorted(block_keys, target_keys), len(block_keys) - 1)
    found = block_keys[pos] == target_keys
    pos, owner = pos[found], target_owner[found]
    sizes = block_sizes[pos]
    if sizes.sum() == 0:
        return None
    target_rows = np.repeat(owner, sizes)
    weights = np.repeat(block_weights[pos], sizes)
    first = np.repeat(block_starts[pos] - np.r_[0, np.cumsum(sizes)[:-1]], sizes)
    source1_rows = source1_by_key[first + np.arange(sizes.sum())]
    base = len(source1_by_key) + 1
    pair_code = target_rows.astype(np.int64) * base + source1_rows
    unique_pairs, inverse = np.unique(pair_code, return_inverse=True)
    score = np.bincount(inverse, weights=weights).astype(np.float32)
    shared = np.bincount(inverse).astype(np.int16)
    target = (unique_pairs // base).astype(np.int64)
    source1 = (unique_pairs % base).astype(np.int64)
    order = np.lexsort((-score, target))
    target, source1, score, shared = target[order], source1[order], score[order], shared[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(target)) + 1]
    group_size = np.diff(np.r_[group_start, len(target)])
    rank = np.arange(len(target)) - np.repeat(group_start, group_size)
    best = np.repeat(score[group_start], group_size)
    hits = np.repeat(group_size, group_size)
    keep = (rank < top_k) & ((rank == 0) | (score >= MIN_SCORE_RATIO * best))
    return pd.DataFrame({"target": target[keep], "source1": source1[keep],
                         "key_score": score[keep], "shared_keys": shared[keep],
                         "key_rank": (rank[keep] + 1).astype(np.int16),
                         "best_key_score": best[keep], "source1_hits": hits[keep].astype(np.int32)})


def find_candidates(source1, targets, source1_country, target_country, top_k=5,
                    chunk=1_000_000, log=print):
    s1_country = np.array([c.strip().lower() for c in source1_country], dtype=object)
    tgt_country = np.array([c.strip().lower() for c in target_country], dtype=object)
    parts = []
    for country in sorted(set(s1_country.tolist())):
        s1_rows = np.flatnonzero(s1_country == country)
        if country:
            tgt_rows = np.flatnonzero((tgt_country == country) | (tgt_country == ""))
        else:
            tgt_rows = np.arange(len(tgt_country))
        if len(s1_rows) == 0 or len(tgt_rows) == 0:
            continue
        s1_cores = [source1["core"][i] for i in s1_rows]
        s1_addrs = [source1["addr"][i] for i in s1_rows]
        name_df, addr_df = document_frequency(s1_cores), document_frequency(s1_addrs)
        with Pool(WORKERS, initializer=_init_worker, initargs=(name_df, addr_df)) as pool:
            started = time.time()
            s1_hashes, s1_owner = compute_keys(pool, s1_cores, s1_addrs,
                                               [source1["nums"][i] for i in s1_rows], True)
            order = np.argsort(s1_hashes, kind="stable")
            s1_hashes, s1_owner = s1_hashes[order], s1_owner[order]
            block_keys, block_starts, block_sizes = np.unique(s1_hashes, return_index=True,
                                                              return_counts=True)
            small = block_sizes <= MAX_BLOCK_SIZE
            block_keys, block_starts, block_sizes = block_keys[small], block_starts[small], block_sizes[small]
            block_weights = np.log(len(s1_rows) / block_sizes).astype(np.float32)
            log(f"[block] {country or '<empty>'}: S1={len(s1_rows)} targets={len(tgt_rows)} "
                f"S1 keys={len(s1_hashes)} blocks kept={len(block_keys)} ({small.mean():.3f}) "
                f"{time.time() - started:.0f}s")
            for start in range(0, len(tgt_rows), chunk):
                batch = tgt_rows[start:start + chunk]
                started = time.time()
                tgt_hashes, tgt_owner = compute_keys(pool, [targets["core"][i] for i in batch],
                                                     [targets["addr"][i] for i in batch],
                                                     [targets["nums"][i] for i in batch], False)
                keyed = time.time()
                cuts = np.searchsorted(tgt_owner, np.linspace(0, len(batch), JOIN_THREADS + 1).astype(np.int64))
                slices = [(tgt_hashes[a:b], tgt_owner[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
                del tgt_hashes, tgt_owner
                with ThreadPoolExecutor(JOIN_THREADS) as executor:
                    results = list(executor.map(
                        lambda part: join_top_candidates(block_keys, block_starts, block_sizes,
                                                         block_weights, s1_owner, part[0], part[1],
                                                         top_k), slices))
                del slices
                log(f"[block]   targets {start}-{start + len(batch)}: keys {keyed - started:.0f}s "
                    f"join {time.time() - keyed:.0f}s")
                for result in results:
                    if result is None:
                        continue
                    result["target"] = batch[result["target"].to_numpy(dtype=np.intp)]
                    result["source1"] = s1_rows[result["source1"].to_numpy(dtype=np.intp)]
                    parts.append(result)
    if not parts:
        return pd.DataFrame({c: np.zeros(0, np.int64) for c in CANDIDATE_COLUMNS})
    return pd.concat(parts, ignore_index=True)
