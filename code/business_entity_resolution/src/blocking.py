import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

MAX_BLOCK = 40
RARE_DF = 25
WORKERS = max(1, (os.cpu_count() or 2) - 1)
JOIN_THREADS = 8
MIN_RATIO = 0.4
_G = {}


def _pairs(toks):
    out = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            out.append(a + "|" + b if a < b else b + "|" + a)
    return out


WIDE = dict(name=5, addr=8, nums=3, hn_tokens=99, x_name=3, x_addr=6, y_name=3)
NARROW = dict(name=4, addr=4, nums=2, hn_tokens=3, x_name=2, x_addr=2, y_name=2)


def segment(tok, vocab, min_piece=3):
    n = len(tok)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + min_piece, n + 1):
            piece = tok[i:j]
            if piece in vocab and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + [piece]
    return best[n] if best[n] and len(best[n]) >= 2 else []


def name_tokens(core, ndf):
    toks = set()
    for t in core.split():
        if t in ndf:
            toks.add(t)
        elif len(t) >= 7:
            toks.update(segment(t, ndf))
    return sorted(toks, key=lambda t: (ndf[t], t))


def record_keys(core, addr, nums, ndf, adf, w):
    nt = name_tokens(core, ndf)
    at = sorted({t for t in addr.split() if t in adf and not t.isdigit()}, key=lambda t: (adf[t], t))
    nums = nums.split()
    keys = []
    if len(nt) == 1:
        keys.append("M" + nt[0])
    keys += ["N" + p for p in _pairs(nt[:w["name"]])]
    keys += ["U" + t for t in nt if ndf[t] <= RARE_DF]
    keys += ["A" + p for p in _pairs(at[:w["addr"]])]
    for n in nums[:w["nums"]]:
        keys += ["H" + n + "|" + t for t in at[:w["hn_tokens"]]]
        keys += ["Y" + t + "|" + n for t in nt[:w["y_name"]]]
    keys += ["X" + a + "|" + b for a in nt[:w["x_name"]] for b in at[:w["x_addr"]]]
    return keys


def _init(ndf, adf):
    _G["ndf"], _G["adf"] = ndf, adf


def _work(args):
    cores, addrs, nums, offset, wide = args
    ndf, adf = _G["ndf"], _G["adf"]
    w = WIDE if wide else NARROW
    ks, own = [], []
    for i, (c, a, n) in enumerate(zip(cores, addrs, nums)):
        k = record_keys(c, a, n, ndf, adf, w)
        ks += k
        own += [offset + i] * len(k)
    h = pd.util.hash_array(np.array(ks, dtype=object)) if ks else np.zeros(0, np.uint64)
    return h, np.array(own, dtype=np.int64)


def _keys(pool, cores, addrs, nums, wide, size=40000):
    jobs = [(cores[i:i + size], addrs[i:i + size], nums[i:i + size], i, wide)
            for i in range(0, len(cores), size)]
    res = pool.map(_work, jobs)
    return (np.concatenate([r[0] for r in res]) if res else np.zeros(0, np.uint64),
            np.concatenate([r[1] for r in res]) if res else np.zeros(0, np.int64))


def _df(strings):
    from collections import Counter
    c = Counter()
    for s in strings:
        c.update(set(s.split()))
    return dict(c)


def _topk_join(ukeys, starts, counts, idf, s1_of, tk, towner, k):
    if len(ukeys) == 0 or len(tk) == 0:
        return None
    pos = np.searchsorted(ukeys, tk)
    pos = np.minimum(pos, len(ukeys) - 1)
    ok = ukeys[pos] == tk
    pos, own = pos[ok], towner[ok]
    c = counts[pos]
    rows = np.repeat(own, c)
    w = np.repeat(idf[pos], c)
    first = np.repeat(starts[pos] - np.r_[0, np.cumsum(c)[:-1]], c)
    s1 = s1_of[first + np.arange(c.sum())]
    if len(rows) == 0:
        return None
    code = rows.astype(np.int64) * (len(s1_of) + 1) + s1
    uc, inv = np.unique(code, return_inverse=True)
    score = np.bincount(inv, weights=w).astype(np.float32)
    nkeys = np.bincount(inv).astype(np.int16)
    t = (uc // (len(s1_of) + 1)).astype(np.int64)
    q = (uc % (len(s1_of) + 1)).astype(np.int64)
    order = np.lexsort((-score, t))
    t, q, score, nkeys = t[order], q[order], score[order], nkeys[order]
    grp_start = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
    rank = np.arange(len(t)) - np.repeat(grp_start, np.diff(np.r_[grp_start, len(t)]))
    best = np.repeat(score[grp_start], np.diff(np.r_[grp_start, len(t)]))
    n_hit = np.repeat(np.diff(np.r_[grp_start, len(t)]), np.diff(np.r_[grp_start, len(t)]))
    keep = (rank < k) & ((rank == 0) | (score >= MIN_RATIO * best))
    return pd.DataFrame({"t": t[keep], "q": q[keep], "kscore": score[keep], "nkeys": nkeys[keep],
                         "krank": (rank[keep] + 1).astype(np.int16), "kbest": best[keep],
                         "khits": n_hit[keep].astype(np.int32)})


def retrieve(n1, nt, c1, ct, k=5, chunk=1_000_000, log=print):
    c1 = np.asarray(c1, dtype=object)
    ct = np.asarray(ct, dtype=object)
    keys1 = [x.strip().lower() for x in c1]
    keyst = np.array([x.strip().lower() for x in ct], dtype=object)
    keys1 = np.array(keys1, dtype=object)
    parts = []
    for country in sorted(set(keys1.tolist())):
        qi = np.flatnonzero(keys1 == country)
        ti = np.flatnonzero((keyst == country) | (keyst == "")) if country else np.arange(len(keyst))
        if len(qi) == 0 or len(ti) == 0:
            continue
        cores1 = [n1["core"][i] for i in qi]
        addrs1 = [n1["addr"][i] for i in qi]
        ndf, adf = _df(cores1), _df(addrs1)
        with Pool(WORKERS, initializer=_init, initargs=(ndf, adf)) as pool:
            t0 = time.time()
            h1, o1 = _keys(pool, cores1, addrs1, [n1["nums"][i] for i in qi], True)
            order = np.argsort(h1, kind="stable")
            h1, o1 = h1[order], o1[order]
            ukeys, starts, counts = np.unique(h1, return_index=True, return_counts=True)
            keep = counts <= MAX_BLOCK
            ukeys, starts, counts = ukeys[keep], starts[keep], counts[keep]
            idf = np.log(len(qi) / counts).astype(np.float32)
            log(f"[block] {country or '<empty>'}: S1={len(qi)} targets={len(ti)} "
                f"S1 keys={len(h1)} blocks kept={len(ukeys)} ({keep.mean():.3f}) {time.time() - t0:.0f}s")
            for s in range(0, len(ti), chunk):
                tsel = ti[s:s + chunk]
                t0 = time.time()
                ht, ot = _keys(pool, [nt["core"][i] for i in tsel], [nt["addr"][i] for i in tsel],
                               [nt["nums"][i] for i in tsel], False)
                t1 = time.time()
                cuts = np.searchsorted(ot, np.linspace(0, len(tsel), JOIN_THREADS + 1).astype(np.int64))
                slices = [(ht[a:b], ot[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
                del ht, ot
                with ThreadPoolExecutor(JOIN_THREADS) as ex:
                    rs = list(ex.map(lambda sl: _topk_join(ukeys, starts, counts, idf, o1,
                                                           sl[0], sl[1], k), slices))
                del slices
                log(f"[block]   targets {s}-{s + len(tsel)}: keys {t1 - t0:.0f}s "
                    f"join {time.time() - t1:.0f}s")
                for r in rs:
                    if r is None:
                        continue
                    r["t"] = tsel[r.t.values]
                    r["q"] = qi[r.q.values]
                    parts.append(r)
    if not parts:
        return pd.DataFrame(columns=["t", "q", "kscore", "nkeys", "krank", "kbest", "khits"])
    return pd.concat(parts, ignore_index=True)
