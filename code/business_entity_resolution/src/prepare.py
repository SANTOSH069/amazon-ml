import os
from multiprocessing import Pool

import numpy as np

from normalize import clean_address, clean_name

WORKERS = max(1, (os.cpu_count() or 2) - 1)
_worker_state = {}


def _init_worker(indic_words, address_synonyms):
    _worker_state["indic_words"] = indic_words
    _worker_state["address_synonyms"] = address_synonyms


def _normalize_chunk(chunk):
    names, addrs = chunk
    indic_words = _worker_state["indic_words"]
    synonyms = _worker_state["address_synonyms"]
    full_names, cores, flags, clean_addrs, numbers = [], [], [], [], []
    for name, addr in zip(names, addrs):
        full, core, is_alias, is_website, is_indic = clean_name(name, indic_words)
        full_names.append(full)
        cores.append(core)
        flags.append(is_alias | (is_website << 1) | (is_indic << 2))
        words, house_numbers = clean_address(addr, indic_words, synonyms)
        clean_addrs.append(" ".join(words))
        numbers.append(" ".join(house_numbers))
    return full_names, cores, np.array(flags, dtype=np.int8), clean_addrs, numbers


def _chunks(names, addrs, size=50000):
    for i in range(0, len(names), size):
        yield names[i:i + size], addrs[i:i + size]


def normalize_records(names, addrs, indic_words, address_synonyms):
    with Pool(WORKERS, initializer=_init_worker, initargs=(indic_words, address_synonyms)) as pool:
        parts = pool.map(_normalize_chunk, _chunks(names, addrs))
    records = {"name": [], "core": [], "addr": [], "nums": []}
    flags = []
    for full_names, cores, chunk_flags, clean_addrs, numbers in parts:
        records["name"] += full_names
        records["core"] += cores
        records["addr"] += clean_addrs
        records["nums"] += numbers
        flags.append(chunk_flags)
    flags = np.concatenate(flags) if flags else np.zeros(0, np.int8)
    records["alias"] = (flags & 1) > 0
    records["web"] = (flags & 2) > 0
    records["script"] = (flags & 4) > 0
    return records


def address_words_only(addrs, indic_words):
    return [clean_address(a, indic_words, {})[0] for a in addrs]
