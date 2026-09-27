# Business Entity Resolution: reproduction guide

This folder contains everything needed to regenerate both submission files from the challenge data:

- `output/matching_results.tsv`: the final matches (the leaderboard file)
- `output/candidate_pairs.tsv`: the exact candidate set the matching model scores

The pipeline runs in four stages: **data → blocking → matching → output**. One command runs all of them, and each stage prints its own progress lines so you can follow along. The rest of this guide explains how to set it up, run it, and check what you should see at each stage.

The pipeline uses only the provided training and test files. It makes no external lookups, loads no pretrained models, and calls no APIs.

---

## 1. Setup

**Tested environment:** Python 3.14.0 on Windows 11, 16 CPU cores, 16 GB RAM. Any recent 64-bit Python 3 should work. The code uses all available cores, so fewer cores means a longer run.

```bash
pip install -r requirements.txt
```

| Package | Version | Used for | License |
|---|---|---|---|
| numpy | 2.4.0 | arrays, sorted-index join | BSD |
| pandas | 3.0.3 | tables | BSD |
| pyarrow | 25.0.1 | fast TSV reading (about 1 s per 10M rows) | Apache 2.0 |
| rapidfuzz | 3.14.6 | multithreaded string similarity (C++) | MIT |
| lightgbm | 4.7.0 | candidate filter and matcher models | MIT |
| scikit-learn | 1.8.0 | required by LightGBM's classifier interface | BSD |
| scipy | 1.17.1 | required by LightGBM | BSD |

---

## 2. Where the data goes

Place the challenge `dataset/` folder next to `code/`, the same way it sits in `student_resource/`:

```
<submission root>/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── output/                          ← written by the pipeline
├── code/business_entity_resolution/
│   ├── src/                         ← all source code
│   ├── README.md                    ← this file
│   └── requirements.txt
└── Documentation_template.md
```

You don't need to configure any paths. The script searches the current directory, this folder, and the submission root for `train_source1.tsv`, and writes to an `output/` folder next to the dataset folder it finds. To use other locations, pass `--data-dir` and `--out-dir` (see section 5).

---

## 3. Reproduce everything: one command

From `code/business_entity_resolution/`:

```bash
python src/main.py
```

On the test machine this takes about **34 minutes**: 19 minutes to train and 15 minutes to resolve the test set. When it finishes, it checks both output files against the submission rules and prints `[validate] PASS`.

To check the files with the official validator as well, run this from the submission root:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`utils/validate_submission.py` ships with `student_resource/` and isn't part of this folder.

---

## 4. What happens at each stage, and what you should see

The same four stages run twice: first on **train**, to learn the model, and then on **test**, to produce the submission. Every stage prints a line that starts with a tag in brackets. The numbers below are from the reference run.

### Stage 1: Data: loading and normalisation
**Code:** `io_utils.py`, `translit.py`, `normalize.py`, `prepare.py`

1. **Load and profile.** The script reads the seven TSV files with pyarrow, keeping every value as a string, so a business named "NA" stays a name. It then prints a profile of each file: rows, countries, empty fields, and singleton rate.
2. **Split and learn dictionaries (training only).** The training Source 1 entities are split into three disjoint groups:
   - **fit (60%):** learns the dictionaries below and trains both models
   - **tune (20%):** chooses the cut-offs and the threshold
   - **eval (20%):** never used for anything except the final score

   Two dictionaries are learned from the fit group's labelled pairs:
   - **Indic word → Latin word**, for example `लिमिटेड → limited`. This comes from name pairs whose words line up one to one.
   - **Target address spelling → Source 1 spelling**, for example `texas → tx`, `keralam → kerala`, `tg → telangana`, `aenue → ave`.
3. **Normalise every record in parallel.** This takes about 1 minute for 12.5M records.
   - Indic scripts are converted to Latin letters and accents are removed.
   - For aliases such as `X dba Y`, only `Y` is kept, and websites such as `acme.com` are unwrapped.
   - Leetspeak is fixed (`5ecure → secure`) and abbreviations are made uniform (`Pvt Ltd → private limited`, `Rd → rd`).
   - Honorifics, filler words and duplicate words are removed.
   - Each record ends up with a name, a *core* name (without legal and generic words), an address, and its house numbers.

```
[fit] S1=2206821 targets=10320219 pairs=7638365 singletons=0.056 target-in-several-S1=0.0000 cross-country=0.0000
[fit] learned 1312 Indic words, 132 address synonyms 29s
[norm] S1=2206821 targets=10320219 64s
```

### Stage 2: Blocking: candidate generation
**Code:** `blocking.py` (stage 2a), then `features.py` and `resolver.py` (stage 2b)

**2a. Key index (per country).** Each record emits short keys made from its rarest words:
- pairs of name words
- pairs of address words
- house number × street word
- name word × house number
- name word × address word

A key shared by more than 40 Source 1 records is dropped as too common. A Source 2/3 record looks up its keys in a sorted index of Source 1 keys. This is a hash join, not an all-pairs comparison. Each Source 2/3 record keeps its best Source 1 match, plus up to 4 more that score at least 40% of the best.

**2b. Learned filter.** A small LightGBM model (150 trees) scores each pair from cheap features. Its cut-off is chosen on the tune group to keep 99.5% of the true pairs. The pairs that survive make up **`candidate_pairs.tsv`**.

```
[block] india: S1=883188 targets=4133346 S1 keys=50279570 blocks kept=15105234 (0.992) 61s
[block] pool=16921294 (7.67/S1) recall=0.9704 390s
[filter] tau=0.0618 kept 9134550 (4.14/S1) recall of pool=0.9950 overall=0.9655 109s
```

On the test set this gives **4.72 candidates per Source 1 entity** (8,173,383 pairs). France is blocked as its own country partition.

### Stage 3: Matching
**Code:** `features.py`, `resolver.py`

1. **Features.** 52 features are computed per candidate pair:
   - name and address similarity (rapidfuzz ratio, token-set, token-sort, partial, Jaro-Winkler)
   - house-number agreement
   - *frequency* features that catch look-alike decoy businesses
   - blocking scores and flags

   No feature uses the country label.
2. **Matcher.** A LightGBM model (400 trees) gives the probability that a pair is the same business.
3. **Decision.** Each Source 2/3 record goes only to its most likely Source 1 entity, because in the training labels no record matches two entities. The match is kept if the probability is at least **0.69**. That threshold is chosen on the tune group to maximise macro F0.5, singletons included.

```
[features] (9134550, 52) 245s
[match] top features: name_ratio, name_tset, extra_tok_max_tdf, addr_tsort, addr_tset, kscore, ...
[tune] threshold=0.69 tune F0.5=0.9752
[eval] held-out S1=441521  macro F0.5=0.9756  candidate recall=0.9655  candidates/S1=4.14
[fit] saved .../model/resolver.pkl  (1135s)
```

The `[eval]` line is the honest score, measured on Source 1 entities that no stage ever trained or tuned on.

### Stage 4: Output
**Code:** `main.py`, `io_utils.py`

The test set goes through stages 1–3 using the saved dictionaries, models and thresholds. Then both files are written, with exactly one row per Source 1 entity, an empty list for "no match", no duplicate IDs, and only Source 2/3 IDs. Finally the files are checked against every submission rule, including that each match is also a candidate.

```
[test] S1=1732544 candidates=8173383 (4.72/S1) matches=5775491 (3.33/S1) non-empty=1627709
[validate] PASS
done in 2028s -> .../output/matching_results.tsv, .../output/candidate_pairs.tsv
```

---

## 5. Running parts of the pipeline

| Command | What it does | Time |
|---|---|---|
| `python src/main.py` | train, then resolve test (the full reproduction) | about 34 min |
| `python src/main.py --mode fit` | train only; prints the held-out score and saves `model/resolver.pkl` | about 19 min |
| `python src/main.py --mode predict` | resolve test with the saved model | about 15 min |

| Option | Default | Meaning |
|---|---|---|
| `--data-dir DIR` | auto-detected | folder that contains `train/` and `test/` |
| `--out-dir DIR` | `output/` next to the dataset folder | where the two TSVs are written |
| `--model PATH` | `model/resolver.pkl` in this folder | where the trained model is saved or loaded |
| `--k N` | 5 | stage-2a: maximum Source 1 candidates kept per Source 2/3 record |
| `--target-recall R` | 0.995 | stage-2b: share of the stage-2a true pairs the filter must keep |
| `--seed N` | 42 | seed for the fit/tune/eval split and the models; runs are reproducible |

---

## 6. Code map

| File | Stage | Responsibility |
|---|---|---|
| `src/main.py` | all | command line: finds the data, runs fit and/or predict, writes and validates the outputs |
| `src/io_utils.py` | 1, 4 | fast TSV reading, data profile, output writing, rule checks |
| `src/translit.py` | 1 | rule-based conversion of Indic scripts to Latin letters (one table for all 9 Unicode blocks) |
| `src/normalize.py` | 1 | name and address cleaning, and learning the two dictionaries |
| `src/prepare.py` | 1 | runs normalisation in parallel over a whole split |
| `src/blocking.py` | 2a | key generation, key index, threaded join, top-K per Source 2/3 record |
| `src/features.py` | 2b, 3 | cheap filter features, full matcher features, frequency features |
| `src/resolver.py` | 2b, 3 | fit/tune/eval protocol, both LightGBM models, threshold, one-match-per-record decision, prediction |

---

## 7. Resources and troubleshooting
- **Memory:** a full run fits in 16 GB of RAM. With less memory, close other programs first. The largest steps work in chunks of 1M records or 5M pairs.
- **Time:** blocking and feature computation use every core. Expect roughly proportional slow-downs on smaller machines.
- **"No dataset found":** pass `--data-dir <folder containing train/ and test/>`.
- **Determinism:** the seed fixes the group split and the models. Small differences in the last decimal place are possible across platforms because of multithreading.
- **Methodology:** see `Documentation_template.md` at the submission root for the full methodology, metrics and error analysis.
