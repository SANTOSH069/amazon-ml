# Business Entity Resolution: How to Reproduce

This folder contains the full pipeline that produces the two submission files:

- `output/matching_results.tsv`: the final matches (the leaderboard file)
- `output/candidate_pairs.tsv`: the exact candidate set that the matching model scores

The pipeline has four stages: data, blocking, matching and output. A single command runs all of them, and each stage prints progress lines so you can follow the run. This guide covers setup, how to run it, and what to expect at each stage.

Only the provided training and test files are used. There are no external lookups, no pretrained models and no API calls.

## 1. Setup

Tested with Python 3.14.0 on Windows 11, 16 CPU cores and 16 GB RAM. Any recent 64-bit Python 3 should work. The code uses all available cores, so fewer cores means a longer run.

```bash
pip install -r requirements.txt
```

| Package | Version | Used for | License |
|---|---|---|---|
| numpy | 2.4.0 | arrays and the sorted-index join | BSD |
| pandas | 3.0.3 | tables | BSD |
| pyarrow | 25.0.1 | fast TSV reading | Apache 2.0 |
| rapidfuzz | 3.14.6 | multithreaded string similarity | MIT |
| lightgbm | 4.7.0 | candidate filter and matching model | MIT |
| scikit-learn | 1.8.0 | required by the LightGBM classifier interface | BSD |
| scipy | 1.17.1 | required by LightGBM | BSD |

## 2. Where the data goes

Place the challenge `dataset/` folder next to `code/`, the same way it is laid out in `student_resource/`:

```
<submission root>/
├── dataset/
│   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── output/                          (written by the pipeline)
├── code/business_entity_resolution/
│   ├── src/                         (all source code)
│   ├── README.md                    (this file)
│   └── requirements.txt
└── Documentation_template.md
```

No paths need to be configured. The script searches the current directory, this folder and the submission root for `train_source1.tsv`, and writes to an `output/` folder next to the dataset folder it finds. Use `--data-dir` and `--out-dir` to choose other locations.

## 3. Reproduce everything with one command

From `code/business_entity_resolution/`:

```bash
python src/main.py
```

On the reference machine this takes about 41 minutes: roughly 23 to train and 18 to resolve the test set. At the end the output files are checked against the submission rules, and the run prints `[validate] PASS`.

To also run the official validator (it ships with `student_resource/`), run this from the submission root:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## 4. What each stage does and what you should see

The four stages run twice: first on the training data to learn the model, then on the test data to produce the submission. Each stage prints lines starting with a tag in square brackets. The numbers below come from the reference run.

### Stage 1: Data (loading and normalisation)

Files: `io_utils.py`, `translit.py`, `normalize.py`, `prepare.py`

1. Load: The seven TSV files are read with pyarrow, keeping every value as a string, so a business called "NA" stays a name. A short profile of each file is printed.
2. Split and learn (training only): Source 1 entities are split into three groups:
   - fit (60%): trains the models
   - tune (20%): chooses the cutoffs
   - eval (20%): only for the final score

   Two dictionaries are learned from the fit group's labelled pairs:
   - Indic word to Latin word, for example `लिमिटेड` to `limited`
   - target address spelling to Source 1 spelling, for example `texas` to `tx`, `keralam` to `kerala`, `aenue` to `ave`
3. Normalise: All records are cleaned in parallel:
   - Indic scripts are converted to Latin letters and accents removed.
   - Aliases (`X dba Y` keeps `Y`) and website names are resolved.
   - Leetspeak (`5ecure` becomes `secure`) and abbreviations (`Pvt Ltd` becomes `private limited`) are standardised.
   - Honorifics, filler words and repeated words are dropped.

   Each record ends up with a name, a core name (without legal and generic words), an address and its house numbers.

```
[fit] S1=2206821 targets=10320219 pairs=7638365 singletons=0.056 target-in-several-S1=0.0000 cross-country=0.0000
[fit] learned 1312 Indic words, 132 address synonyms
[norm] S1=2206821 targets=10320219
```

### Stage 2: Blocking (candidate generation)

Files: `blocking.py`, then `features.py` and `resolver.py`

- 2a. Key index, built per country: Every record produces short keys from its rarest words:
  - pairs of name words
  - pairs of address words
  - house number with street word
  - name word with house number
  - name word with address word

  Keys shared by more than 40 Source 1 records are dropped as too common. Each Source 2/3 record looks up its keys in a sorted index of Source 1 keys (a hash join, not an all-pairs comparison). It keeps its best Source 1 match, plus up to four more that score at least 40% of the best.
- 2b. Candidate filter: A small LightGBM model (150 trees) scores each pair from cheap features. Its cutoff is set on the tune group to keep 99.5% of the true pairs. The pairs that pass become `candidate_pairs.tsv`.

```
[block] pool=16920627 (7.67/S1) recall=0.9704
[filter] cutoff=0.0618 kept 9134550 (4.14/S1) recall of pool=0.9950 overall=0.9655
```

On the test set this gives 4.72 candidates per Source 1 entity. France is handled as its own country partition.

### Stage 3: Matching

Files: `features.py`, `resolver.py`

1. Features: 52 features are computed per candidate pair:
   - name and address similarity (ratio, token set, token sort, partial ratio, Jaro-Winkler)
   - house number agreement
   - frequency features that detect look-alike businesses
   - blocking scores and simple flags

   The country label is never used as a feature.
2. Model: A LightGBM model with 400 trees gives the probability that a pair is the same business.
3. Decision:
   - Each Source 2/3 record goes to at most one Source 1 entity, its most likely one. In the training labels, no record ever belongs to two entities.
   - The match is kept if its probability reaches the threshold chosen on the tune group to maximise macro F0.5, singletons included.

```
[tune] threshold=0.69 tune F0.5=0.9752
[eval] held-out S1=441521  macro F0.5=0.9756  candidate recall=0.9655  candidates/S1=4.14
```

The `[eval]` line is the honest score, measured on Source 1 entities that were never used for training or tuning.

### Stage 4: Output

Files: `main.py`, `io_utils.py`

- Resolve the test set: Stages 1 to 3 run on the test data using the saved dictionaries, models and threshold.
- Write both files: Each has exactly one row per Source 1 entity, an empty list for "no match", no duplicate IDs, and only Source 2/3 IDs.
- Check against the rules: The rules include that every match also appears in the candidate list.

```
[test] S1=1732544 candidates=8176824 (4.72/S1) matches=5776256 (3.33/S1) non-empty=1627740
[validate] PASS
```

## 5. Running parts of the pipeline

| Command | What it does |
|---|---|
| `python src/main.py` | trains, then resolves the test set (full reproduction) |
| `python src/main.py --mode fit` | trains only; prints the held-out score and saves `model/resolver.pkl` |
| `python src/main.py --mode predict` | resolves the test set with the saved model (about 18 minutes) |

| Option | Default | Meaning |
|---|---|---|
| `--data-dir DIR` | found automatically | folder containing `train/` and `test/` |
| `--out-dir DIR` | `output/` next to the dataset | where the two TSV files are written |
| `--model PATH` | `model/resolver.pkl` | where the trained model is saved or loaded |
| `--top-k N` | 5 | blocking: most Source 1 candidates kept per Source 2/3 record |
| `--target-recall R` | 0.995 | candidate filter: share of true pairs it must keep |
| `--seed N` | 42 | seed for the group split and the models |

## 6. Code map

| File | Stage | Responsibility |
|---|---|---|
| `src/main.py` | all | command line: finds the data, runs training and/or prediction, writes and checks the outputs |
| `src/io_utils.py` | 1, 4 | TSV reading, data profile, writing the outputs, rule checks |
| `src/translit.py` | 1 | converts Indic scripts to Latin letters (one table covers all nine Unicode blocks) |
| `src/normalize.py` | 1 | name and address cleaning; learns the two dictionaries |
| `src/prepare.py` | 1 | runs the cleaning in parallel over a whole split |
| `src/blocking.py` | 2a | key building, key index, threaded join, top candidates per Source 2/3 record |
| `src/features.py` | 2b, 3 | candidate filter features, matching features, frequency features |
| `src/resolver.py` | 2b, 3 | training protocol, both LightGBM models, threshold choice, one-match-per-record decision, prediction |

## 7. Resources and troubleshooting

- Memory: A full run fits in 16 GB of RAM. With less memory, close other programs first; the heaviest steps already work in chunks.
- Time: Blocking and feature computation use every core. Expect roughly proportional slowdowns on smaller machines.
- "No dataset found": Pass `--data-dir` with the folder that contains `train/` and `test/`.
- Repeatability: The seed fixes the group split and the models. The last decimal place of a score can vary slightly across machines because of multithreading.
- Methodology: See `Documentation_template.md` in the submission root for the full methodology, results and error analysis.
