import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import blocking
from features import add_q_context, cheap_features, full_features, signature_stats
from normalize import learn_addr_synonyms, learn_native_dict
from prepare import addr_tokens_plain, normalize_records
from translit import has_indic

BETA2 = 0.25


def macro_f05(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(p + r > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean()) if len(f) else 0.0


def score_submission(pred, truth):
    tp, npred, ntrue = [], [], []
    for k, t in truth.items():
        p = set(pred.get(k, []))
        tp.append(len(p & set(t)))
        npred.append(len(p))
        ntrue.append(len(t))
    return macro_f05(tp, npred, ntrue)


def _lgb(n_estimators, seed):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.08, num_leaves=127,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=seed, verbose=-1)


def _predict(model, X, chunk=4_000_000):
    return np.concatenate([model.predict_proba(X.iloc[i:i + chunk])[:, 1]
                           for i in range(0, len(X), chunk)]) if len(X) else np.zeros(0)


def _in_chunks(fn, pairs, chunk=5_000_000):
    if len(pairs) == 0:
        return fn(pairs)
    return pd.concat([fn(pairs.iloc[i:i + chunk]) for i in range(0, len(pairs), chunk)],
                     ignore_index=True)


def assign(t, q, p):
    if len(t) == 0:
        return t, q, p
    order = np.lexsort((-p, t))
    t, q, p = t[order], q[order], p[order]
    first = np.r_[True, t[1:] != t[:-1]]
    return t[first], q[first], p[first]


class Resolver:
    def __init__(self, k=5, target_recall=0.995, seed=42, log=print):
        self.k = k
        self.target_recall = target_recall
        self.seed = seed
        self.log = log

    def _normalize(self, s1, s2, s3):
        t0 = time.time()
        tg = pd.concat([s2, s3], ignore_index=True)
        n1 = normalize_records(s1.business_name.tolist(), s1.business_address.tolist(),
                               self.native, self.native, self.syn)
        nt = normalize_records(tg.business_name.tolist(), tg.business_address.tolist(),
                               self.native, self.native, self.syn)
        t_src = np.r_[np.full(len(s2), 2, np.int8), np.full(len(s3), 3, np.int8)]
        self.log(f"[norm] S1={len(s1)} targets={len(tg)} {time.time() - t0:.0f}s")
        return n1, nt, tg, t_src

    def _candidates(self, n1, nt, s1, tg, t_src):
        t0 = time.time()
        pool = blocking.retrieve(n1, nt, s1.country.values, tg.country.values, k=self.k,
                                 log=self.log)
        self.log(f"[block] pool={len(pool)} ({len(pool) / max(len(s1), 1):.2f}/S1) "
                 f"{time.time() - t0:.0f}s")
        t0 = time.time()
        cheap = _in_chunks(lambda p: cheap_features(p, n1, nt), pool)
        pool["p2"] = _predict(self.stage2, cheap[self.cheap_cols])
        keep = pool.p2.values >= self.tau
        self.log(f"[filter] kept {keep.sum()} of {len(pool)} ({keep.sum() / max(len(s1), 1):.2f}/S1) "
                 f"{time.time() - t0:.0f}s")
        return pool, cheap, keep

    def fit(self, s1, s2, s3, gt):
        rng = np.random.default_rng(self.seed)
        grp = rng.choice(3, size=len(s1), p=[0.6, 0.2, 0.2])
        tg = pd.concat([s2, s3], ignore_index=True)
        ids = gt.matched_entity_ids.str.split(",")
        ex = pd.DataFrame({"s1": gt.source1_entity_id.values, "t": ids}).explode("t")
        ex = ex[ex.t.fillna("") != ""]
        s1i = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
        ti = pd.Series(np.arange(len(tg)), index=tg.entity_id.values)
        ex = ex[ex.s1.isin(s1i.index) & ex.t.isin(ti.index)]
        pq, pt = s1i[ex.s1.values].values, ti[ex.t.values].values
        owner = np.full(len(tg), -1, np.int64)
        owner[pt] = pq
        ntrue = np.bincount(pq, minlength=len(s1))
        multi = pd.Series(pt).duplicated().mean()
        cross = (s1.country.values[pq] != tg.country.values[pt]).mean() if len(pq) else 0
        self.log(f"[fit] S1={len(s1)} targets={len(tg)} pairs={len(pq)} "
                 f"singletons={np.mean(ntrue == 0):.3f} target-in-several-S1={multi:.4f} "
                 f"cross-country={cross:.4f}")

        t0 = time.time()
        fm = grp[pq] == 0
        fq, ft = pq[fm], pt[fm]
        tn = tg.business_name.values
        ind = np.fromiter((has_indic(x) for x in tn[ft]), bool, len(ft))
        self.native = learn_native_dict(s1.business_name.values[fq[ind]], tn[ft[ind]])
        samp = rng.choice(len(fq), size=min(600_000, len(fq)), replace=False)
        self.syn = learn_addr_synonyms(
            addr_tokens_plain(s1.business_address.values[fq[samp]], self.native),
            addr_tokens_plain(tg.business_address.values[ft[samp]], self.native))
        self.log(f"[fit] learned {len(self.native)} Indic words, {len(self.syn)} address "
                 f"synonyms {time.time() - t0:.0f}s")

        n1, nt, _, t_src = self._normalize(s1, s2, s3)

        t0 = time.time()
        pool = blocking.retrieve(n1, nt, s1.country.values, tg.country.values, k=self.k,
                                 log=self.log)
        y = owner[pool.t.values] == pool.q.values
        g = grp[pool.q.values]
        self.log(f"[block] pool={len(pool)} ({len(pool) / len(s1):.2f}/S1) "
                 f"recall={y.sum() / len(pq):.4f} {time.time() - t0:.0f}s")

        t0 = time.time()
        cheap = _in_chunks(lambda p: cheap_features(p, n1, nt), pool)
        self.cheap_cols = list(cheap.columns)
        fit_idx = np.flatnonzero(g == 0)
        fit_idx = rng.choice(fit_idx, size=min(len(fit_idx), 8_000_000), replace=False)
        self.stage2 = _lgb(150, self.seed).fit(cheap.iloc[fit_idx], y[fit_idx])
        p2 = _predict(self.stage2, cheap)
        tune = g == 1
        pos = np.sort(p2[tune & y])
        self.tau = float(pos[int((1 - self.target_recall) * len(pos))]) if len(pos) else 0.5
        keep = p2 >= self.tau
        self.log(f"[filter] tau={self.tau:.4f} kept {keep.sum()} ({keep.sum() / len(s1):.2f}/S1) "
                 f"recall of pool={y[keep].sum() / max(y.sum(), 1):.4f} "
                 f"overall={y[keep].sum() / len(pq):.4f} {time.time() - t0:.0f}s")

        t0 = time.time()
        cand = pool[keep].reset_index(drop=True)
        cheap = cheap[keep].reset_index(drop=True)
        y, g = y[keep], g[keep]
        del pool
        gc.collect()
        X = self._full(cand, cheap, n1, nt, t_src)
        del cheap
        gc.collect()
        self.cols = list(X.columns)
        fit_idx = np.flatnonzero(g == 0)
        fit_idx = rng.choice(fit_idx, size=min(len(fit_idx), 8_000_000), replace=False)
        self.model = _lgb(400, self.seed).fit(X.iloc[fit_idx], y[fit_idx])
        p = _predict(self.model, X)
        self.log(f"[match] features {X.shape} trained {time.time() - t0:.0f}s")
        imp = sorted(zip(self.model.feature_importances_, self.cols), reverse=True)
        self.log("[match] top features: " + ", ".join(f"{c}={v}" for v, c in imp[:12]))

        t, q, pb = assign(cand.t.values, cand.q.values, p)
        yb = owner[t] == q
        best = (-1.0, 0.5)
        for thr in np.round(np.arange(0.05, 0.99, 0.01), 2):
            s = self._group_f(t, q, pb, yb, thr, grp, ntrue, 1)
            if s > best[0]:
                best = (s, float(thr))
        self.thr = best[1]
        f_eval = self._group_f(t, q, pb, yb, self.thr, grp, ntrue, 2)
        ev = grp == 2
        ev_pairs = ev[cand.q.values]
        n_ev_pos = ntrue[ev].sum()
        self.log(f"[tune] threshold={self.thr} tune F0.5={best[0]:.4f}")
        self.log(f"[eval] held-out S1={ev.sum()}  macro F0.5={f_eval:.4f}  "
                 f"candidate recall={y[ev_pairs].sum() / max(n_ev_pos, 1):.4f}  "
                 f"candidates/S1={ev_pairs.sum() / ev.sum():.2f}")
        self.eval_report = dict(f05=f_eval, thr=self.thr, tau=self.tau,
                                recall=float(y[ev_pairs].sum() / max(n_ev_pos, 1)),
                                cand_per_s1=float(ev_pairs.sum() / ev.sum()))
        return self

    @staticmethod
    def _group_f(t, q, p, y, thr, grp, ntrue, which):
        m = (p >= thr) & (grp[q] == which)
        n = len(ntrue)
        tp = np.bincount(q[m], weights=y[m].astype(float), minlength=n)
        npred = np.bincount(q[m], minlength=n)
        sel = grp == which
        return macro_f05(tp[sel], npred[sel], ntrue[sel])

    def _full(self, cand, cheap, n1, nt, t_src, chunk=5_000_000):
        t0 = time.time()
        st = signature_stats(n1, nt)
        X = pd.concat([full_features(cand.iloc[i:i + chunk], n1, nt, t_src,
                                     cheap=cheap.iloc[i:i + chunk].reset_index(drop=True),
                                     stats=st)
                       for i in range(0, max(len(cand), 1), chunk)], ignore_index=True)
        X = add_q_context(X, cand)
        self.log(f"[features] {X.shape} {time.time() - t0:.0f}s")
        return X

    def predict(self, s1, s2, s3):
        normalized = self._normalize(s1, s2, s3)
        if normalized is None:
            raise RuntimeError("Normalization did not return the expected values")
        n1, nt, tg, t_src = normalized
        pool, cheap, keep = self._candidates(n1, nt, s1, tg, t_src)
        cand = pool[keep].reset_index(drop=True)
        cheap = cheap[keep].reset_index(drop=True)
        del pool
        gc.collect()
        t0 = time.time()
        X = self._full(cand, cheap, n1, nt, t_src)
        del cheap
        gc.collect()
        p = _predict(self.model, X[self.cols])
        t, q, pb = assign(cand.t.values, cand.q.values, p)
        m = pb >= self.thr
        self.log(f"[match] scored {len(cand)} pairs, {m.sum()} matches {time.time() - t0:.0f}s")

        s1_ids = s1.entity_id.values
        t_ids = tg.entity_id.values
        cand_lists = _group_lists(cand.q.values, t_ids[cand.t.values], s1_ids)
        order = np.lexsort((-pb[m], q[m]))
        match_lists = _group_lists(q[m][order], t_ids[t[m][order]], s1_ids)
        return cand_lists, match_lists


def _group_lists(q, ids, s1_ids):
    out = {e: [] for e in s1_ids}
    if len(q) == 0:
        return out
    order = np.argsort(q, kind="stable")
    q, ids = q[order], ids[order]
    bounds = np.flatnonzero(np.r_[True, q[1:] != q[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[s1_ids[q[a]]] = ids[a:b].tolist()
    return out
