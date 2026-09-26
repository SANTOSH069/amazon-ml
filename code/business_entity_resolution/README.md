# Business Entity Resolution

The pipeline normalises the text, finds candidates with key blocking, filters them with a small LightGBM model, scores them with a second LightGBM model, and gives each Source 2/3 record to at most one Source 1 entity.
It uses only the provided data. It makes no external lookups and loads no pretrained models.

## Run

```bash
pip install -r requirements.txt
python src/main.py                     # fit on train, then resolve test (about 35 min)
python src/main.py --mode fit          # train only: prints the held-out score, saves model/resolver.pkl
python src/main.py --mode predict      # resolve test with the saved model (about 15 min)
python src/main.py --data-dir <dir with train/ and test/> --out-dir <dir>
```

The dataset is found automatically. The script searches the current directory, then the project, for `train_source1.tsv`.
Outputs are `output/matching_results.tsv` and `output/candidate_pairs.tsv` (by default next to the dataset folder). Both are checked against the submission rules at the end of the run.

These timings were measured on 16 cores with 16 GB of RAM, which was enough for the full run.

## Layout

| file | role |
|---|---|
| `src/translit.py` | rule-based romanisation of Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) |
| `src/normalize.py` | name/address normalisation, plus two dictionaries learned from training pairs: Indic word → Latin word, and target address spelling → Source 1 spelling |
| `src/prepare.py` | parallel normalisation of a whole split |
| `src/blocking.py` | stage 1: hashed-key inverted index, top-K Source 1 records per target, partitioned by country |
| `src/features.py` | stage-2 cheap features and the full matcher features (rapidfuzz similarities and frequency features) |
| `src/resolver.py` | fit (fit/tune/eval split of Source 1), filter, matcher, threshold, prediction |
| `src/io_utils.py` | fast TSV reading (pyarrow), writing, rule checks |
| `src/main.py` | CLI |

`candidate_pairs.tsv` is exactly the set of pairs the matcher scores, so every match is also a candidate.
