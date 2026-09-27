import gc
import time
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import find_candidates
from features import add_source1_context, filter_features, frequency_tables, matcher_features
from normalize import learn_address_synonyms, learn_indic_words
from prepare import address_words_only, normalize_records
from translit import has_indic

BETA_SQUARED = 0.25
FIT, TUNE, EVAL = 0, 1, 2


def macro_f05(true_pos, predicted, actual):
    true_pos, predicted, actual = (np.asarray(x, float) for x in (true_pos, predicted, actual))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, true_pos / predicted, 0.0)
        recall = np.where(actual > 0, true_pos / actual, 0.0)
        score = np.where(precision + recall > 0,
                         (1 + BETA_SQUARED) * precision * recall / (BETA_SQUARED * precision + recall), 0.0)
    score = np.where((actual == 0) & (predicted == 0), 1.0, score)
    return float(score.mean()) if len(score) else 0.0


def score_submission(predictions, truth):
    true_pos, predicted, actual = [], [], []
    for entity, true_ids in truth.items():
        guessed = set(predictions.get(entity, []))
        true_pos.append(len(guessed & set(true_ids)))
        predicted.append(len(guessed))
        actual.append(len(true_ids))
    return macro_f05(true_pos, predicted, actual)


def new_lightgbm(trees, seed):
    return lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.08, num_leaves=127,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                              random_state=seed, verbose=-1)


def predict_in_chunks(model, feats, chunk=4_000_000):
    classes = list(model.classes_)
    if len(feats) == 0 or True not in classes:
        return np.zeros(len(feats))
    if len(classes) == 1:
        return np.ones(len(feats))
    column = classes.index(True)
    return np.concatenate([model.predict_proba(feats.iloc[i:i + chunk])[:, column]
                           for i in range(0, len(feats), chunk)])


def apply_in_chunks(func, pairs, chunk=5_000_000):
    if len(pairs) == 0:
        return func(pairs)
    return pd.concat([func(pairs.iloc[i:i + chunk]) for i in range(0, len(pairs), chunk)],
                     ignore_index=True)


def best_source1_per_target(target, source1, prob):
    if len(target) == 0:
        return target, source1, prob
    order = np.lexsort((-prob, target))
    target, source1, prob = target[order], source1[order], prob[order]
    first = np.r_[True, target[1:] != target[:-1]]
    return target[first], source1[first], prob[first]


def group_f05(target, source1, prob, correct, threshold, group, actual, which):
    accepted = (prob >= threshold) & (group[source1] == which)
    n = len(actual)
    true_pos = np.bincount(source1[accepted], weights=correct[accepted].astype(float), minlength=n)
    predicted = np.bincount(source1[accepted], minlength=n)
    members = group == which
    return macro_f05(true_pos[members], predicted[members], actual[members])


def ids_by_source1(source1_rows, ids, source1_ids):
    lists = {entity: [] for entity in source1_ids}
    if len(source1_rows) == 0:
        return lists
    order = np.argsort(source1_rows, kind="stable")
    source1_rows, ids = source1_rows[order], ids[order]
    bounds = np.flatnonzero(np.r_[True, source1_rows[1:] != source1_rows[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        lists[source1_ids[source1_rows[a]]] = ids[a:b].tolist()
    return lists


class Resolver:
    def __init__(self, top_k=5, target_recall=0.998, seed=42, log=print):
        self.top_k = top_k
        self.target_recall = target_recall
        self.seed = seed
        self.log = log

    def normalize(self, s1, s2, s3):
        started = time.time()
        targets_raw = pd.concat([s2, s3], ignore_index=True)
        source1 = normalize_records(s1.business_name.tolist(), s1.business_address.tolist(),
                                    self.indic_words, self.address_synonyms)
        targets = normalize_records(targets_raw.business_name.tolist(),
                                    targets_raw.business_address.tolist(),
                                    self.indic_words, self.address_synonyms)
        target_source = np.r_[np.full(len(s2), 2, np.int8), np.full(len(s3), 3, np.int8)]
        self.log(f"[norm] S1={len(s1)} targets={len(targets_raw)} {time.time() - started:.0f}s")
        return source1, targets, targets_raw, target_source

    def build_matcher_features(self, candidates, filter_feats, source1, targets, target_source,
                               chunk=5_000_000):
        started = time.time()
        tables = frequency_tables(source1, targets)
        feats = pd.concat([
            matcher_features(candidates.iloc[i:i + chunk], source1, targets, target_source,
                             filter_feats=filter_feats.iloc[i:i + chunk].reset_index(drop=True),
                             tables=tables)
            for i in range(0, max(len(candidates), 1), chunk)], ignore_index=True)
        feats = add_source1_context(feats, candidates)
        self.log(f"[features] {feats.shape} {time.time() - started:.0f}s")
        return feats

    def fit(self, s1, s2, s3, ground_truth):
        rng = np.random.default_rng(self.seed)
        group = rng.choice(3, size=len(s1), p=[0.6, 0.2, 0.2])
        targets_raw = pd.concat([s2, s3], ignore_index=True)
        pairs = pd.DataFrame({"source1": ground_truth.source1_entity_id.values,
                              "target": ground_truth.matched_entity_ids.str.split(",")}).explode("target")
        pairs = pairs[pairs.target.fillna("") != ""]
        s1_row = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
        target_row = pd.Series(np.arange(len(targets_raw)), index=targets_raw.entity_id.values)
        pairs = pairs[pairs.source1.isin(s1_row.index) & pairs.target.isin(target_row.index)]
        true_s1, true_target = s1_row[pairs.source1.values].values, target_row[pairs.target.values].values
        owner = np.full(len(targets_raw), -1, np.int64)
        owner[true_target] = true_s1
        actual = np.bincount(true_s1, minlength=len(s1))
        shared_targets = pd.Series(true_target).duplicated().mean()
        cross_country = ((s1.country.values[true_s1] != targets_raw.country.values[true_target]).mean()
                         if len(true_s1) else 0)
        self.log(f"[fit] S1={len(s1)} targets={len(targets_raw)} pairs={len(true_s1)} "
                 f"singletons={np.mean(actual == 0):.3f} target-in-several-S1={shared_targets:.4f} "
                 f"cross-country={cross_country:.4f}")

        started = time.time()
        in_fit = group[true_s1] == FIT
        fit_s1, fit_target = true_s1[in_fit], true_target[in_fit]
        target_names = targets_raw.business_name.values
        indic = np.fromiter((has_indic(x) for x in target_names[fit_target]), bool, len(fit_target))
        self.indic_words = learn_indic_words(s1.business_name.values[fit_s1[indic]],
                                             target_names[fit_target[indic]])
        sample = rng.choice(len(fit_s1), size=min(600_000, len(fit_s1)), replace=False)
        self.address_synonyms = learn_address_synonyms(
            address_words_only(s1.business_address.values[fit_s1[sample]], self.indic_words),
            address_words_only(targets_raw.business_address.values[fit_target[sample]], self.indic_words))
        self.log(f"[fit] learned {len(self.indic_words)} Indic words, "
                 f"{len(self.address_synonyms)} address synonyms {time.time() - started:.0f}s")

        source1, targets, _, target_source = self.normalize(s1, s2, s3)

        started = time.time()
        pool = find_candidates(source1, targets, s1.country.values, targets_raw.country.values,
                               top_k=self.top_k, log=self.log)
        is_match = owner[pool.target.values] == pool.source1.values
        pair_group = group[pool.source1.values]
        self.log(f"[block] pool={len(pool)} ({len(pool) / len(s1):.2f}/S1) "
                 f"recall={is_match.sum() / len(true_s1):.4f} {time.time() - started:.0f}s")

        started = time.time()
        filter_feats = apply_in_chunks(lambda p: filter_features(p, source1, targets), pool)
        self.filter_columns = list(filter_feats.columns)
        train_rows = np.flatnonzero(pair_group == FIT)
        train_rows = rng.choice(train_rows, size=min(len(train_rows), 8_000_000), replace=False)
        self.candidate_filter = new_lightgbm(200, self.seed).fit(filter_feats.iloc[train_rows],
                                                                 is_match[train_rows])
        filter_prob = predict_in_chunks(self.candidate_filter, filter_feats)
        tune_positive = np.sort(filter_prob[(pair_group == TUNE) & is_match])
        self.filter_cutoff = (float(tune_positive[int((1 - self.target_recall) * len(tune_positive))])
                              if len(tune_positive) else 0.5)
        keep = filter_prob >= self.filter_cutoff
        self.log(f"[filter] cutoff={self.filter_cutoff:.4f} kept {keep.sum()} "
                 f"({keep.sum() / len(s1):.2f}/S1) recall of pool={is_match[keep].sum() / max(is_match.sum(), 1):.4f} "
                 f"overall={is_match[keep].sum() / len(true_s1):.4f} {time.time() - started:.0f}s")

        started = time.time()
        candidates = pool[keep].reset_index(drop=True)
        filter_feats = filter_feats[keep].reset_index(drop=True)
        is_match, pair_group = is_match[keep], pair_group[keep]
        del pool
        gc.collect()
        feats = self.build_matcher_features(candidates, filter_feats, source1, targets, target_source)
        del filter_feats
        gc.collect()
        self.matcher_columns = list(feats.columns)
        train_rows = np.flatnonzero(pair_group == FIT)
        train_rows = rng.choice(train_rows, size=min(len(train_rows), 8_000_000), replace=False)
        self.matcher = new_lightgbm(600, self.seed).fit(feats.iloc[train_rows], is_match[train_rows])
        prob = predict_in_chunks(self.matcher, feats)
        self.log(f"[match] features {feats.shape} trained {time.time() - started:.0f}s")
        ranked = sorted(zip(self.matcher.feature_importances_, self.matcher_columns), reverse=True)
        self.log("[match] top features: " + ", ".join(f"{name}={value}" for value, name in ranked[:12]))

        best_target, best_s1, best_prob = best_source1_per_target(
            candidates.target.values, candidates.source1.values, prob)
        correct = owner[best_target] == best_s1
        best = (-1.0, 0.5)
        for threshold in np.round(np.arange(0.05, 0.99, 0.005), 3):
            score = group_f05(best_target, best_s1, best_prob, correct, threshold, group, actual, TUNE)
            if score > best[0]:
                best = (score, float(threshold))
        self.threshold = best[1]
        eval_score = group_f05(best_target, best_s1, best_prob, correct, self.threshold, group, actual, EVAL)
        in_eval = group == EVAL
        eval_pairs = in_eval[candidates.source1.values]
        eval_recall = is_match[eval_pairs].sum() / max(actual[in_eval].sum(), 1)
        self.log(f"[tune] threshold={self.threshold} tune F0.5={best[0]:.4f}")
        self.log(f"[eval] held-out S1={in_eval.sum()}  macro F0.5={eval_score:.4f}  "
                 f"candidate recall={eval_recall:.4f}  candidates/S1={eval_pairs.sum() / in_eval.sum():.2f}")
        self.eval_report = dict(f05=eval_score, threshold=self.threshold, filter_cutoff=self.filter_cutoff,
                                candidate_recall=float(eval_recall),
                                candidates_per_s1=float(eval_pairs.sum() / in_eval.sum()))
        return self

    def predict(self, s1, s2, s3):
        source1, targets, targets_raw, target_source = cast(Any, self.normalize(s1, s2, s3))
        started = time.time()
        pool = find_candidates(source1, targets, s1.country.values, targets_raw.country.values,
                               top_k=self.top_k, log=self.log)
        self.log(f"[block] pool={len(pool)} ({len(pool) / max(len(s1), 1):.2f}/S1) "
                 f"{time.time() - started:.0f}s")
        started = time.time()
        filter_feats = apply_in_chunks(lambda p: filter_features(p, source1, targets), pool)
        keep = predict_in_chunks(self.candidate_filter, filter_feats[self.filter_columns]) >= self.filter_cutoff
        self.log(f"[filter] kept {keep.sum()} of {len(pool)} ({keep.sum() / max(len(s1), 1):.2f}/S1) "
                 f"{time.time() - started:.0f}s")
        candidates = pool[keep].reset_index(drop=True)
        filter_feats = filter_feats[keep].reset_index(drop=True)
        del pool
        gc.collect()
        started = time.time()
        feats = self.build_matcher_features(candidates, filter_feats, source1, targets, target_source)
        del filter_feats
        gc.collect()
        prob = predict_in_chunks(self.matcher, feats[self.matcher_columns])
        best_target, best_s1, best_prob = best_source1_per_target(
            candidates.target.values, candidates.source1.values, prob)
        accepted = best_prob >= self.threshold
        self.log(f"[match] scored {len(candidates)} pairs, {accepted.sum()} matches "
                 f"{time.time() - started:.0f}s")

        source1_ids = s1.entity_id.values
        target_ids = targets_raw.entity_id.values
        candidate_lists = ids_by_source1(candidates.source1.values,
                                         target_ids[candidates.target.values], source1_ids)
        order = np.lexsort((-best_prob[accepted], best_s1[accepted]))
        match_lists = ids_by_source1(best_s1[accepted][order],
                                     target_ids[best_target[accepted][order]], source1_ids)
        return candidate_lists, match_lists
