import os
from multiprocessing import Pool

import numpy as np

from normalize import addr_parts, name_parts

WORKERS = max(1, (os.cpu_count() or 2) - 1)
_G = {}


def _init(native, addr_native, syn):
    _G["native"], _G["addr_native"], _G["syn"] = native, addr_native, syn


def _work(chunk):
    names, addrs = chunk
    native, addr_native, syn = _G["native"], _G["addr_native"], _G["syn"]
    nm, core, flags, ad, nums = [], [], [], [], []
    for n, a in zip(names, addrs):
        x = name_parts(n, native)
        nm.append(x[0])
        core.append(x[1])
        flags.append(x[2] | (x[3] << 1) | (x[4] << 2))
        t, d = addr_parts(a, addr_native, syn)
        ad.append(" ".join(t))
        nums.append(" ".join(d))
    return nm, core, np.array(flags, dtype=np.int8), ad, nums


def _chunks(names, addrs, size=50000):
    for i in range(0, len(names), size):
        yield names[i:i + size], addrs[i:i + size]


def normalize_records(names, addrs, native, addr_native, syn, pool=None):
    own = pool is None
    if own:
        pool = Pool(WORKERS, initializer=_init, initargs=(native, addr_native, syn))
    try:
        parts = pool.map(_work, _chunks(names, addrs))
    finally:
        if own:
            pool.close()
            pool.join()
    out = {"name": [], "core": [], "addr": [], "nums": []}
    flags = []
    for nm, core, fl, ad, nums in parts:
        out["name"] += nm
        out["core"] += core
        out["addr"] += ad
        out["nums"] += nums
        flags.append(fl)
    fl = np.concatenate(flags) if flags else np.zeros(0, np.int8)
    out["alias"], out["web"], out["script"] = (fl & 1) > 0, (fl & 2) > 0, (fl & 4) > 0
    return out


def addr_tokens_plain(addrs, addr_native):
    return [addr_parts(a, addr_native, {})[0] for a in addrs]
