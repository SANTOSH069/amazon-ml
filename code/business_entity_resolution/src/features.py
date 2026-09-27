from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

BLOCKING_COLUMNS = ["key_score", "shared_keys", "key_rank", "best_key_score", "source1_hits"]


def pairwise(left, right, scorer):
    return process.cpdist(left, right, scorer=scorer, workers=-1, dtype=np.float32)


def number_features(source1_numbers, target_numbers):
    first_match = np.empty(len(source1_numbers), np.float32)
    jaccard = np.empty(len(source1_numbers), np.float32)
    for i, (left, right) in enumerate(zip(source1_numbers, target_numbers)):
        if not left or not right:
            first_match[i] = jaccard[i] = -1.0
            continue
        left_list, right_list = left.split(), right.split()
        first_match[i] = 1.0 if left_list[0] in right_list else 0.0
        left_set, right_set = set(left_list), set(right_list)
        jaccard[i] = len(left_set & right_set) / len(left_set | right_set)
    return first_match, jaccard


def joined_name_coverage(source1_cores, target_cores):
    coverage = np.empty(len(source1_cores), np.float32)
    for i, (source1_core, target_core) in enumerate(zip(source1_cores, target_cores)):
        words = [w for w in source1_core.split() if len(w) >= 3]
        if not words:
            coverage[i] = -1.0
            continue
        joined = target_core.replace(" ", "")
        total = sum(len(w) for w in words)
        coverage[i] = sum(len(w) for w in words if w in joined) / total
    return coverage


def filter_features(pairs, source1, targets):
    s1_idx, tgt_idx = pairs.source1.values, pairs.target.values
    s1_core = [source1["core"][i] for i in s1_idx]
    tgt_core = [targets["core"][i] for i in tgt_idx]
    s1_addr = [source1["addr"][i] for i in s1_idx]
    tgt_addr = [targets["addr"][i] for i in tgt_idx]
    feats = pd.DataFrame({c: pairs[c].values for c in BLOCKING_COLUMNS})
    feats["key_score_ratio"] = pairs.key_score.values / np.maximum(pairs.best_key_score.values, 1e-6)
    feats["core_token_set"] = pairwise(s1_core, tgt_core, fuzz.token_set_ratio)
    feats["core_ratio"] = pairwise(s1_core, tgt_core, fuzz.ratio)
    feats["address_token_set"] = pairwise(s1_addr, tgt_addr, fuzz.token_set_ratio)
    feats["first_number_match"], feats["number_jaccard"] = number_features(
        [source1["nums"][i] for i in s1_idx], [targets["nums"][i] for i in tgt_idx])
    feats["target_is_website"] = targets["web"][tgt_idx].astype(np.int8)
    feats["target_is_indic_script"] = targets["script"][tgt_idx].astype(np.int8)
    feats["target_has_no_address"] = np.array([not a for a in tgt_addr], dtype=np.int8)
    return feats


def add_source1_context(feats, pairs):
    frame = pd.DataFrame({"source1": pairs.source1.values, "score": pairs.key_score.values})
    feats["source1_candidate_count"] = frame.groupby("source1").source1.transform("size").values.astype(np.int32)
    feats["source1_candidate_rank"] = frame.groupby("source1").score.rank(
        ascending=False, method="min").values.astype(np.float32)
    return feats


def record_signatures(records):
    name_sigs, address_sigs = [], []
    for core, addr, nums in zip(records["core"], records["addr"], records["nums"]):
        name_sigs.append(" ".join(sorted(set(core.split()))))
        signature = ""
        if nums:
            words = addr.split()
            house = nums.split()[0]
            for i, word in enumerate(words):
                if word.lstrip("0") == house or (word[:1].isdigit() and house in word):
                    street = next((w for w in words[i + 1:] if not w[:1].isdigit()), "")
                    signature = house + "|" + street
                    break
        address_sigs.append(signature)
    return name_sigs, address_sigs


def hash_strings(strings):
    return pd.util.hash_array(np.array(strings, dtype=object))


def frequency_tables(source1, targets):
    tables = {}
    for side, records in (("source1", source1), ("target", targets)):
        name_sigs, address_sigs = record_signatures(records)
        name_hash, address_hash = hash_strings(name_sigs), hash_strings(address_sigs)
        combined_hash = hash_strings([n + "#" + a for n, a in zip(name_sigs, address_sigs)])
        address_hash[np.array([not a for a in address_sigs], dtype=bool)] = 0
        tables[("name", side)], tables[("address", side)] = name_hash, address_hash
        tables[("combined", side)] = combined_hash
    for kind in ("name", "address", "combined"):
        s1_hash, tgt_hash = tables[(kind, "source1")], tables[(kind, "target")]
        unique, inverse = np.unique(np.concatenate([s1_hash, tgt_hash]), return_inverse=True)
        tables[(kind, "unique")] = unique
        tables[(kind, "count_source1")] = np.bincount(inverse[:len(s1_hash)], minlength=len(unique))
        tables[(kind, "count_target")] = np.bincount(inverse[len(s1_hash):], minlength=len(unique))
    tables["word_df_source1"] = Counter(w for c in source1["core"] for w in set(c.split()))
    tables["word_df_target"] = Counter(w for c in targets["core"] for w in set(c.split()))
    return tables


def signature_counts(tables, kind, hashes):
    unique = tables[(kind, "unique")]
    if len(unique) == 0:
        return np.zeros(len(hashes), np.float32), np.zeros(len(hashes), np.float32)
    pos = np.minimum(np.searchsorted(unique, hashes), len(unique) - 1)
    found = unique[pos] == hashes
    in_source1 = np.where(found, tables[(kind, "count_source1")][pos], 0)
    in_targets = np.where(found, tables[(kind, "count_target")][pos], 0)
    return in_source1.astype(np.float32), in_targets.astype(np.float32)


def frequency_features(pairs, source1, targets, tables):
    s1_idx, tgt_idx = pairs.source1.values, pairs.target.values
    feats = {}
    for kind in ("name", "address", "combined"):
        s1_hash = tables[(kind, "source1")][s1_idx]
        tgt_hash = tables[(kind, "target")][tgt_idx]
        feats[f"{kind}_signature_equal"] = (s1_hash == tgt_hash).astype(np.int8)
        feats[f"target_{kind}_in_source1"], feats[f"target_{kind}_in_targets"] = \
            signature_counts(tables, kind, tgt_hash)
        feats[f"source1_{kind}_in_source1"], feats[f"source1_{kind}_in_targets"] = \
            signature_counts(tables, kind, s1_hash)
        if kind == "address":
            missing = tgt_hash == 0
            feats["target_address_in_source1"][missing] = -1
            feats["target_address_in_targets"][missing] = -1
    df_source1, df_target = tables["word_df_source1"], tables["word_df_target"]
    extra_target_freq = np.empty(len(s1_idx), np.float32)
    extra_source1_freq = np.empty(len(s1_idx), np.float32)
    extra_count = np.empty(len(s1_idx), np.int8)
    missing_count = np.empty(len(s1_idx), np.int8)
    for i, (a, b) in enumerate(zip(s1_idx.tolist(), tgt_idx.tolist())):
        s1_words = set(source1["core"][a].split())
        tgt_words = set(targets["core"][b].split())
        extra = tgt_words - s1_words
        extra_count[i] = min(len(extra), 100)
        missing_count[i] = min(len(s1_words - tgt_words), 100)
        if extra:
            extra_target_freq[i] = max(df_target.get(w, 0) for w in extra)
            extra_source1_freq[i] = min(df_source1.get(w, 0) for w in extra)
        else:
            extra_target_freq[i] = extra_source1_freq[i] = -1
    feats["extra_word_target_frequency"] = extra_target_freq
    feats["extra_word_source1_frequency"] = extra_source1_freq
    feats["extra_word_count"] = extra_count
    feats["missing_word_count"] = missing_count
    return pd.DataFrame(feats)


def matcher_features(pairs, source1, targets, target_source, filter_feats=None, tables=None):
    s1_idx, tgt_idx = pairs.source1.values, pairs.target.values
    feats = filter_features(pairs, source1, targets) if filter_feats is None else filter_feats.copy()
    s1_core = [source1["core"][i] for i in s1_idx]
    tgt_core = [targets["core"][i] for i in tgt_idx]
    s1_name = [source1["name"][i] for i in s1_idx]
    tgt_name = [targets["name"][i] for i in tgt_idx]
    s1_addr = [source1["addr"][i] for i in s1_idx]
    tgt_addr = [targets["addr"][i] for i in tgt_idx]
    feats["core_token_sort"] = pairwise(s1_core, tgt_core, fuzz.token_sort_ratio)
    feats["core_partial"] = pairwise(s1_core, tgt_core, fuzz.partial_ratio)
    feats["core_jaro_winkler"] = pairwise(s1_core, tgt_core, JaroWinkler.normalized_similarity)
    feats["name_ratio"] = pairwise(s1_name, tgt_name, fuzz.ratio)
    feats["name_token_set"] = pairwise(s1_name, tgt_name, fuzz.token_set_ratio)
    s1_joined = [c.replace(" ", "") for c in s1_core]
    tgt_joined = [c.replace(" ", "") for c in tgt_core]
    feats["joined_name_ratio"] = pairwise(s1_joined, tgt_joined, fuzz.ratio)
    feats["joined_name_partial"] = pairwise(s1_joined, tgt_joined, fuzz.partial_ratio)
    feats["joined_name_coverage"] = joined_name_coverage(s1_core, tgt_core)
    feats["address_ratio"] = pairwise(s1_addr, tgt_addr, fuzz.ratio)
    feats["address_token_sort"] = pairwise(s1_addr, tgt_addr, fuzz.token_sort_ratio)
    feats["address_partial"] = pairwise(s1_addr, tgt_addr, fuzz.partial_ratio)
    feats["source1_core_words"] = np.array([len(c.split()) for c in s1_core], np.int8)
    feats["target_core_words"] = np.array([len(c.split()) for c in tgt_core], np.int8)
    feats["source1_address_words"] = np.array([len(a.split()) for a in s1_addr], np.int16)
    feats["target_address_words"] = np.array([len(a.split()) for a in tgt_addr], np.int16)
    feats["target_has_alias"] = targets["alias"][tgt_idx].astype(np.int8)
    feats["from_source3"] = (target_source[tgt_idx] == 3).astype(np.int8)
    if tables is not None:
        freq = frequency_features(pairs, source1, targets, tables)
        for c in freq.columns:
            feats[c] = freq[c].values
    return feats
