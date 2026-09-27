import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

KEY_FEATS = ["kscore", "nkeys", "krank", "kbest", "khits"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _num_feats(na, nb):
    first_eq = np.empty(len(na), np.float32)
    jac = np.empty(len(na), np.float32)
    for i, (x, y) in enumerate(zip(na, nb)):
        if not x or not y:
            first_eq[i] = jac[i] = -1.0
            continue
        sx, sy = x.split(), y.split()
        first_eq[i] = 1.0 if sx[0] in sy else 0.0
        a, b = set(sx), set(sy)
        jac[i] = len(a & b) / len(a | b)
    return first_eq, jac


def _concat_cover(core_a, core_b):
    out = np.empty(len(core_a), np.float32)
    for i, (a, b) in enumerate(zip(core_a, core_b)):
        toks = [t for t in a.split() if len(t) >= 3]
        if not toks:
            out[i] = -1.0
            continue
        bb = b.replace(" ", "")
        tot = sum(len(t) for t in toks)
        out[i] = sum(len(t) for t in toks if t in bb) / tot
    return out


def cheap_features(pairs, n1, nt):
    q, t = pairs.q.values, pairs.t.values
    ca = [n1["core"][i] for i in q]
    cb = [nt["core"][i] for i in t]
    aa = [n1["addr"][i] for i in q]
    ab = [nt["addr"][i] for i in t]
    f = pd.DataFrame({k: pairs[k].values for k in KEY_FEATS})
    f["kratio"] = pairs.kscore.values / np.maximum(pairs.kbest.values, 1e-6)
    f["core_tset"] = _cp(ca, cb, fuzz.token_set_ratio)
    f["core_ratio"] = _cp(ca, cb, fuzz.ratio)
    f["addr_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    f["num_first"], f["num_jac"] = _num_feats([n1["nums"][i] for i in q], [nt["nums"][i] for i in t])
    f["t_web"] = nt["web"][t].astype(np.int8)
    f["t_script"] = nt["script"][t].astype(np.int8)
    f["t_noaddr"] = np.array([not x for x in ab], dtype=np.int8)
    return f


def add_q_context(X, pairs):
    g = pd.DataFrame({"q": pairs.q.values, "s": pairs.kscore.values})
    X["q_ncand"] = g.groupby("q").q.transform("size").values.astype(np.int32)
    X["q_rank"] = g.groupby("q").s.rank(ascending=False, method="min").values.astype(np.float32)
    return X


def _sigs(side):
    nsig, asig = [], []
    for core, addr, nums in zip(side["core"], side["addr"], side["nums"]):
        nsig.append(" ".join(sorted(set(core.split()))))
        sig = ""
        if nums:
            toks = addr.split()
            first = nums.split()[0]
            for i, tk in enumerate(toks):
                if tk.lstrip("0") == first or (tk[:1].isdigit() and first in tk):
                    nxt = next((x for x in toks[i + 1:] if not x[:1].isdigit()), "")
                    sig = first + "|" + nxt
                    break
        asig.append(sig)
    return nsig, asig


def _h(strings):
    return pd.util.hash_array(np.array(strings, dtype=object))


def signature_stats(n1, nt):
    st = {}
    for key, side in (("1", n1), ("t", nt)):
        ns, as_ = _sigs(side)
        hn, ha = _h(ns), _h(as_)
        hc = _h([a + "#" + b for a, b in zip(ns, as_)])
        ha[np.array([not x for x in as_])] = 0
        st["hn" + key], st["ha" + key], st["hc" + key] = hn, ha, hc
    for kind in ("hn", "ha", "hc"):
        both = np.concatenate([st[kind + "1"], st[kind + "t"]])
        u, inv = np.unique(both, return_inverse=True)
        c1 = np.bincount(inv[:len(st[kind + "1"])], minlength=len(u))
        ct = np.bincount(inv[len(st[kind + "1"]):], minlength=len(u))
        st[kind + "_u"], st[kind + "_c1"], st[kind + "_ct"] = u, c1, ct
    from collections import Counter
    st["tok1"] = Counter(t for c in n1["core"] for t in set(c.split()))
    st["tokt"] = Counter(t for c in nt["core"] for t in set(c.split()))
    return st


def _count(st, kind, h):
    u = st[kind + "_u"]
    pos = np.minimum(np.searchsorted(u, h), len(u) - 1)
    ok = u[pos] == h
    c1 = np.where(ok, st[kind + "_c1"][pos], 0)
    ct = np.where(ok, st[kind + "_ct"][pos], 0)
    return c1.astype(np.float32), ct.astype(np.float32)


def freq_features(pairs, n1, nt, st):
    q, t = pairs.q.values, pairs.t.values
    f = {}
    for kind, nm in (("hn", "name"), ("ha", "addr"), ("hc", "combo")):
        hq, ht = st[kind + "1"][q], st[kind + "t"][t]
        f[f"{nm}_sig_eq"] = (hq == ht).astype(np.int8)
        f[f"t_{nm}_s1cnt"], f[f"t_{nm}_tcnt"] = _count(st, kind, ht)
        f[f"q_{nm}_s1cnt"], f[f"q_{nm}_tcnt"] = _count(st, kind, hq)
        if kind == "ha":
            miss = ht == 0
            for c in ("t_addr_s1cnt", "t_addr_tcnt"):
                f[c][miss] = -1
    tok1, tokt = st["tok1"], st["tokt"]
    d_tdf = np.empty(len(q), np.float32)
    d_s1df = np.empty(len(q), np.float32)
    n_extra = np.empty(len(q), np.int8)
    n_missing = np.empty(len(q), np.int8)
    for i, (a, b) in enumerate(zip(q.tolist(), t.tolist())):
        sa, sb = set(n1["core"][a].split()), set(nt["core"][b].split())
        extra = sb - sa
        n_extra[i] = min(len(extra), 100)
        n_missing[i] = min(len(sa - sb), 100)
        if extra:
            d_tdf[i] = max(tokt.get(x, 0) for x in extra)
            d_s1df[i] = min(tok1.get(x, 0) for x in extra)
        else:
            d_tdf[i] = d_s1df[i] = -1
    f["extra_tok_max_tdf"], f["extra_tok_min_s1df"] = d_tdf, d_s1df
    f["n_extra_tok"], f["n_missing_tok"] = n_extra, n_missing
    return pd.DataFrame(f)


def full_features(pairs, n1, nt, t_src, cheap=None, stats=None):
    q, t = pairs.q.values, pairs.t.values
    f = cheap_features(pairs, n1, nt) if cheap is None else cheap.copy()
    ca = [n1["core"][i] for i in q]
    cb = [nt["core"][i] for i in t]
    na = [n1["name"][i] for i in q]
    nb = [nt["name"][i] for i in t]
    aa = [n1["addr"][i] for i in q]
    ab = [nt["addr"][i] for i in t]
    f["core_tsort"] = _cp(ca, cb, fuzz.token_sort_ratio)
    f["core_partial"] = _cp(ca, cb, fuzz.partial_ratio)
    f["core_jw"] = _cp(ca, cb, JaroWinkler.normalized_similarity)
    f["name_ratio"] = _cp(na, nb, fuzz.ratio)
    f["name_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    ca_ns = [x.replace(" ", "") for x in ca]
    cb_ns = [x.replace(" ", "") for x in cb]
    f["nospace_ratio"] = _cp(ca_ns, cb_ns, fuzz.ratio)
    f["nospace_partial"] = _cp(ca_ns, cb_ns, fuzz.partial_ratio)
    f["concat_cover"] = _concat_cover(ca, cb)
    f["addr_ratio"] = _cp(aa, ab, fuzz.ratio)
    f["addr_tsort"] = _cp(aa, ab, fuzz.token_sort_ratio)
    f["addr_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    f["len_core_a"] = np.array([len(x.split()) for x in ca], np.int8)
    f["len_core_b"] = np.array([len(x.split()) for x in cb], np.int8)
    f["len_addr_a"] = np.array([len(x.split()) for x in aa], np.int16)
    f["len_addr_b"] = np.array([len(x.split()) for x in ab], np.int16)
    f["t_alias"] = nt["alias"][t].astype(np.int8)
    f["is_s3"] = (t_src[t] == 3).astype(np.int8)
    if stats is not None:
        fr = freq_features(pairs, n1, nt, stats)
        for c in fr.columns:
            f[c] = fr[c].values
    return f
