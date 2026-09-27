# ML Challenge 2026: Business Entity Resolution Solution

| | |
|---|---|
| **Team Name** | Beyond Bias |
| **Team Members** | Vedha Shree · Venkata Santosh · Sree Venkatanadh · Venkata Thanush |
| **Submission Date** | 27 September 2026 |

---

## 1. Executive Summary

We treat entity resolution as a question asked of every Source 2/3 record: *which Source 1 business is this, if any?* The pipeline has four parts:

1. **Normalisation**, designed around the noise we measured in the training data, including conversion of eight Indic scripts to Latin letters and two dictionaries learned from the labelled pairs.
2. **Blocking** in two steps: a hashed-key inverted index, then a small learned filter. This leaves **4.1 candidates per Source 1 entity** (4.7 on test) and still keeps 96.6% of the true matches.
3. **A LightGBM matcher** with 52 similarity and *frequency* features. The frequency features are the key idea against look-alike decoy businesses.
4. **A decision rule** that gives each Source 2/3 record to at most one Source 1 entity, with a threshold tuned for macro F0.5.

On 441,521 training Source 1 entities that no step ever trained or tuned on, the pipeline scores **macro F0.5 = 0.9756** (precision 0.9955, recall 0.9446). Its running time grows linearly with the data, and it resolves the 1.73M-entity test set in about 15 minutes on a 16-core machine.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled all of the training data (2,206,821 Source 1 records, 10,320,219 Source 2/3 records, 7,638,365 labelled pairs) before designing anything. These findings shaped every later decision.

**Structure of the matches**

| Finding | Value | Design consequence |
|---|---|---|
| Source 2/3 records matching more than one Source 1 entity | **0** of 7.64M | Each Source 2/3 record goes only to its best Source 1 candidate, and retrieval is organised per Source 2/3 record |
| Matches between different countries | **0** | Blocking runs separately per country string (an open set, so France is handled automatically) |
| Matches per Source 1 entity | mean 3.46, max 11 | The candidate set can be small |
| Singletons (Source 1 entities with no match) | 123,247 (5.6%) | Predicting "no match" correctly is worth 1.0 each, so the threshold is tuned with singletons included |
| Source 2/3 records that match nothing | 26.0% | There are many non-matching records to reject, including deliberate decoys |

**Noise in names** (measured on a sample of 300,000 matched pairs)
- **Case and punctuation:** changes in case, spacing and punctuation; junk prefixes (`***`, `>>`, `--`, `#`, `@`, about 0.3% each); stray trailing brackets and full stops.
- **Accents and leetspeak:** accents added to letters (`Ínc`, `Láwrence`), and leetspeak typos (`5ecure`, `J0aquin`, `Danie1`, `lnc`, `C0mpany`).
- **Word-level changes:** words doubled (`Vetsch, Vetsch`), shuffled (`Miller LLC Edp`) or dropped.
- **Legal suffixes:** swapped legal suffixes are the most common single change (`limited` dropped 23,873 times, `llc` 17,353 times).
- **Added words:** generic words (`services`, `center`, `partners`) and honorifics (`shri`, `smt`, `mr`, `dr`).
- **Aliases:** the real name comes *after* the marker, as in `Arcbrixx dba Rays Office`. Markers found: `dba` (1,266 in the sample), `formerly`, `f/k/a`, `trading as`, `d/b/a`, `t/a`, `aka`, `doing business as`, `a/k/a`, `fka`, `formerly known as`.
- **Websites and hashtags:** 4.6% of records (`emerakeystone.com`, `#danieljoaquin`, `wilfordhancock.com`).
- **Invented names:** a completely invented name at the true address (`Keloonyx`).
- **Indic scripts:** 27.9% of Indian Source 2 names and 18.5% of Indian Source 3 names, across Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada and Malayalam.

**Noise in addresses**
- **Abbreviations:** `Road/Rd`, `Street/St`, `Avenue/Ave`, plus typos in them (`aenue`, `stret`, `dirve`).
- **States:** codes vs full names in both directions (`OH ↔ Ohio`, `Kerala ↔ KL ↔ Keralam ↔ കേരളം`).
- **Filler words:** prefixes such as `Door No`, `H.No`, `Plot`, `Unit UNIT`, `PMB`, and city suffixes such as `CDP` and `CITY`.
- **House-number formats:** `00709`, `#4514`, `4514D`, `5001-5003`, `315.`.
- **Missing and reordered parts:** components dropped or reordered, and 3.3–3.4% of Source 2/3 addresses entirely empty.

**Decoys.** Some unmatched Source 2/3 records copy a Source 1 business with one letter changed in a rare word (`rooposh twin company` vs `roopesh twin company`) or a nearby house number (`8216 bowers ln` vs `8203 bowers ln`). With our first model, **79% of false positives** were decoys of this kind.

**The test set** adds France (259,452 Source 1 records), which has no training labels. French records follow the same noise patterns (`R.`/`Rue`, `BD.`, `N°`, `S.A.R.L.`, an inserted `(FRANCE)`, upper-case names), so French street types and legal forms were added to normalisation. The country label is used only as a partition key and never as a model feature.

### 2.2 Solution Strategy

**Approach Type:** Blocking (hashed-key index + learned filter) → gradient-boosted classifier → one-match-per-record assignment

**Core Innovations:**
1. **Asymmetric key blocking.** Source 1 records emit a *wide* key set and Source 2/3 records a *narrow* one. A Source 2/3 record is a noisy subset of its Source 1 record, so its narrow keys fall inside the wide set. This is what made small candidate sets compatible with high recall.
2. **Retrieval per Source 2/3 record.** Because no record matches two Source 1 entities, each Source 2/3 record keeps only its few best Source 1 matches, which bounds the candidate set.
3. **Frequency features against decoys.** A decoy is a real, separate business, so its unusual spelling repeats across several Source 2/3 records while no Source 1 record has it. A random typo in a true match appears once. This group of features raised macro F0.5 from 0.9696 to 0.9755.
4. **Dictionaries learned from the training labels:** Indic word → Latin word (1,312 entries) and target address spelling → Source 1 spelling (132 entries). Both are learned only from the fit group's pairs.

**Validation protocol.** The training Source 1 entities are split at random (seed 42) into three disjoint groups:

| Group | Share | Used for |
|---|---|---|
| fit | 60% | learning the dictionaries and training both LightGBM models (up to 8M pairs each) |
| tune | 20% | choosing the filter cut-off and the decision threshold |
| eval | 20% | nothing except the reported score |

Normalisation and blocking run over the *whole* training split, so held-out entities compete against every other record, exactly as they would on test.

---

## 3. Candidate Generation (Blocking)

Blocking has two steps. Step 2a is an inverted index over hashed keys that returns a short, ranked list for every Source 2/3 record. Step 2b is a small learned filter that trims that list. The output of step 2b is `candidate_pairs.tsv`: exactly the pairs the matcher scores.

### 3.1 Normalisation (feeds the keys)
- **Indic scripts → Latin.** The nine Indic Unicode blocks share Devanagari's layout, so one table converts all of them (`कृष्णा → krishna`, `ಕರ್ನಾಟಕ → karnatak`). Learned dictionary entries take priority (`प्राइवेट → private`, `ਪ੍ਰਾ → private`).
- **Names.** In order:
  1. accents are folded
  2. aliases are resolved (the name after `dba`, `f/k/a`, … is kept)
  3. websites are unwrapped
  4. dotted abbreviations are joined (`L.L.C. → llc`)
  5. `&`/`+` become "and"
  6. leetspeak is repaired
  7. abbreviations are made uniform (`pvt → private`, `ltd → limited`, `corp → corporation`, `sté → societe`, …)
  8. honorifics and duplicate words are removed

  The **core name** also drops legal forms (US, Indian and French: LLC, Pvt Ltd, SARL, SAS, EURL, SCI, …) and generic words.
- **Addresses.**
  - Street types are abbreviated the same way on both sides (English, Indian and French: `rue`, `blvd`, `all`, …).
  - The 132 learned synonyms are applied.
  - Filler words are removed.
  - House numbers are extracted with leading zeros stripped.

### 3.2 Step 2a: key index
- **Rarity.** A word's rarity is how many Source 1 records *in the same country* contain it. Words that no Source 1 record contains can never produce a candidate, so they are skipped.
- **Key types** (each built from a record's rarest words):

  | Key | Built from | Why |
  |---|---|---|
  | N | pairs of the rarest core-name words | robust to word order, dropped words and suffix changes |
  | M / U | a one-word name / a single very rare word | short or unique names |
  | A | pairs of the rarest address words | robust to reordered or dropped components |
  | H | house number × address word | exact location |
  | **Y** | **name word × house number** | separates generic names (`sky estate`, `balaji impex`) that share a city |
  | X | name word × address word | survives a typo in either field |

- **Asymmetric widths.** Source 1 uses up to 5 name words, 8 address words, 3 house numbers, and house number × every address word. Source 2/3 records use 4 name words, 4 address words, 2 house numbers and 3 address words per number.
- **Website names** are split into Source 1 words by a small dynamic program (`pricetotalpacific → price total pacific`).
- **Block purging.** A key shared by more than 40 Source 1 records is dropped. That caps the work per key, so the join grows linearly with the data rather than quadratically.
- **The join.** Keys are 64-bit hashes held in a sorted array. Each Source 2/3 record looks its keys up with `np.searchsorted`, running in 8 threads over chunks of 1M records. A pair's score is the sum of the IDF weights, log(N/df), of the keys it shares.
- **Top-K.** Each Source 2/3 record keeps its best Source 1 match, plus up to 4 more that score at least 40% of the best. On train this rule keeps 99.996% of the true pairs in the top-5 pool at one-third of its size.

**How recall improved.** Every change came from reading the missed true pairs:

| Version | Recall at top-1 | Recall at top-5 |
|---|---|---|
| Keys from each record's own rarest words (symmetric) | 0.9229 | 0.9499 |
| + asymmetric wide/narrow keys | 0.9368 | 0.9608 |
| + name × house-number keys, website-name splitting | **0.9532** | **0.9704** |

### 3.3 Step 2b: learned filter
- **Model:** LightGBM with 150 trees.
- **Inputs:** key score, rank, best score and number of hits; rapidfuzz `token_set_ratio` and `ratio` on the core name and address; house-number agreement; website, script and no-address flags.
- **Cut-off:** p ≥ 0.0618, chosen on the **tune** group to keep 99.5% of the step-2a true pairs.

### 3.4 Blocking numbers

| | Train (all 2.21M Source 1) | Test (1.73M Source 1) |
|---|---|---|
| Step 2a pool | 16,921,294 pairs (7.67 per Source 1), recall 0.9704 | 18,206,268 pairs (10.51 per Source 1) |
| **Final candidates** (`candidate_pairs.tsv`) | **9,134,550 (4.14 per Source 1)** | **8,173,383 (4.72 per Source 1)** |
| **Blocking recall** (true pairs in the candidates) | **0.9655** | no labels |
| Reduction ratio vs all Source 1 × Source 2/3 pairs | 0.99999960 | 0.99999953 |
| Source 1 entities with no candidates | not measured | 22,273 (1.3%) |
| Blocking time (step 2a) | 390 s | 371 s |

The test set has more Source 2/3 records per Source 1 entity (5.75 vs 4.68 in train), so it has more candidates per entity.

### 3.5 How true matches were kept
- Wide/narrow keys, name × house-number keys, website-name splitting, and learned transliteration and synonyms (section 3.2).
- A filter cut-off set by *recall* on a separate group, never by candidate count.
- **What is still missed:** 3.45% of true pairs. About half of these are Source 2/3 records with **no address** whose name is shared by several Source 1 entities. Any key that could find them is shared by too many records, and even a human couldn't tell which entity is meant.

---

## 4. Matching Model

### 4.1 Features (52 per candidate pair)

| Group | Features |
|---|---|
| **Name similarity** | rapidfuzz `ratio` (normalised Indel/Levenshtein), `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and Jaro-Winkler on the full and core name; ratio and partial ratio with spaces removed; **concat cover** (the share of Source 1 core characters found inside a run-together name); core word counts; number of extra and missing words |
| **Address similarity** | `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`; whether the first house number matches; **Jaccard** of the house-number sets; address word counts; missing-address flag |
| **Frequency (decoy detection)** | For the name signature (sorted core words), the address signature (house number + following street word) and both combined: whether the Source 1 and Source 2/3 signatures are equal, and how many Source 1 and Source 2/3 records share each side's signature. Also the highest Source 2/3 frequency and lowest Source 1 frequency of the words the Source 2/3 record *adds* |
| **Blocking context** | key score, rank, best score, number of Source 1 records hit, number of keys shared, score ratio; number of Source 2/3 records proposing this Source 1 entity and this pair's rank among them |
| **Flags** | Source 2 or 3; Indic-script, website, alias; missing address |

- **Country:** no feature uses the country label, which is what lets the model transfer to France.
- **Frequency features use no labels:** they count records in the split being resolved (the test inputs themselves), so they are unsupervised.
- **Most important features** (by split count): `name_ratio`, `name_tset`, **`extra_tok_max_tdf`** (the decoy signal), `addr_tsort`, `addr_tset`, `kscore`, `addr_partial`, `core_jw`, `t_addr_tcnt`, `addr_ratio`.

### 4.2 Model
**Model type:** LightGBM gradient-boosted trees (MIT license). This is not a pretrained model, and it is far below the 8B-parameter limit.

| | Stage-2b filter | Matcher |
|---|---|---|
| Trees | 150 | 400 |
| Settings | learning rate 0.08, 127 leaves, min_child_samples 50, row/column subsampling 0.8, L2 1.0 | same |
| Training data | up to 8M pairs from the fit group | up to 8M candidate pairs from the fit group |

### 4.3 Threshold selection and decision rule
1. **One Source 1 entity per Source 2/3 record.** Each Source 2/3 record keeps only its highest-probability candidate, because in the training labels no record belongs to two entities.
2. **Threshold.** The match is accepted if p ≥ **0.69**, found by a grid search from 0.05 to 0.98 that maximises **macro F0.5 on the tune group**, singletons included.
3. **Result per entity.** A Source 1 entity's matches are the Source 2/3 records that chose it. If none did, it gets an empty list.

---

## 5. Results & Error Analysis

All figures below are for the **final saved model** on the **eval group** (441,521 Source 1 entities and 1,528,407 true pairs), unless marked otherwise.

### 5.1 Headline metrics

| Metric | Value |
|---|---|
| **F_0.5 Score (macro, official metric)** | **0.9756** |
| Precision (pair level) | 0.9955 |
| Recall (pair level) | 0.9446 |
| Blocking recall | 0.9655 |
| Matcher recall on candidates | 0.9784 |
| Source 1 entities predicted exactly right | 83.0% |
| Singletons correctly left empty | 97.9% (24,105 / 24,628) |
| Macro F0.5 on the fit group / tune group | 0.9764 / 0.9752 (so no overfitting) |

### 5.2 Confusion matrix (pair level)

|  | Predicted match | Predicted no-match |
|---|---|---|
| **True match** | **TP 1,443,679** | **FN 84,728** (52,803 missed by blocking + 31,925 rejected by the matcher) |
| **Not a match** | **FP 6,504** | **TN 345,985** (candidate pairs correctly rejected) |

About 10¹² further pairs were never candidates. They are all true negatives, handled by blocking.

### 5.3 Confusion matrix (entity level: any match or none?)

|  | Predicted some match | Predicted empty |
|---|---|---|
| **Has true matches** | 414,063 | 2,830 |
| **True singleton** | 523 | 24,105 |

### 5.4 By country (eval group)

| Country | Source 1 entities | Macro F0.5 | Precision | Recall | Blocking recall | Candidates per Source 1 | Singleton accuracy |
|---|---|---|---|---|---|---|---|
| US | 264,911 | 0.9807 | 0.9963 | 0.9542 | 0.9758 | 4.20 | 0.9799 |
| India | 176,610 | 0.9680 | 0.9944 | 0.9301 | 0.9500 | 4.05 | 0.9770 |
| France (test only) | 259,452 | no labels | | | | | |

India is harder. It has native-script names, long free-form addresses, and many generic names (`balaji impex`). France has no labels, so we sanity-checked the test output: France averages 3.20 matches per entity with 6.1% empty, against 3.26 / 6.3% for India and 3.47 / 5.7% for the US. That is in line with the ~5.6% singleton rate in training.

### 5.5 Precision–recall trade-off
This comes from a development run that used the same pipeline and features but a different seed. Its eval macro F0.5 was 0.9755, against 0.9756 for the saved model.

| Threshold | Precision | Recall | Macro F0.5 |
|---|---|---|---|
| 0.30 | 0.9855 | 0.9556 | 0.9711 |
| 0.50 | 0.9919 | 0.9507 | 0.9748 |
| 0.60 | 0.9939 | 0.9478 | 0.9754 |
| **0.69 (chosen)** | **0.9956** | **0.9443** | **0.9755** |
| 0.80 | 0.9969 | 0.9396 | 0.9747 |
| 0.90 | 0.9983 | 0.9303 | 0.9718 |

The curve is flat near its best point, so small shifts in the threshold cost little.

### 5.6 Errors
- **Common false positives (wrong merges):**
  - **Decoys:** unmatched records that copy a Source 1 business with one changed letter or a nearby house number. These were 79% of false positives before the frequency features, which removed a large share of them.
  - The rest are records with an invented name at an address shared by several businesses.
- **Common false negatives (missed matches):**
  1. **Records with no address and a generic name** (the largest group). They are ambiguous in principle.
  2. **True matches whose house number was also changed.** This carries the same signal as a decoy, so a precision-first threshold rejects some of them.
  3. **Names with heavy typos combined with partial addresses.**
  4. **Blocking misses** (3.45% of true pairs), concentrated in India.

---

## 6. How we applied the "Tips for Success"

**1. "Invest in a strong blocking/candidate generation strategy — it determines the upper bound of your recall."**
- **What we did:** blocking got the most engineering effort. We ran three rounds of diagnosing missed true pairs, which gave asymmetric keys, name × house-number keys and website-name splitting (section 3.2), plus a learned filter tuned by recall.
- **Evidence:** top-5 recall rose from 0.9499 to 0.9704 while the final candidate set fell to **4.14 per Source 1 entity**, only slightly above the 3.46 true matches per entity. Block purging keeps the running time linear. Test blocking takes about 6 minutes for 1.73M × 9.97M records, and the reduction ratio is 0.9999995.

**2. "Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and address matching."**
- **Levenshtein family:** rapidfuzz `ratio` (normalised Indel distance, a Levenshtein variant), `partial_ratio`, token-sort/token-set ratios and Jaro-Winkler, on names, core names and addresses.
- **Jaccard:** on house-number sets. Token-set ratios capture word-overlap similarity.
- **TF-IDF / IDF:** our first prototype used TF-IDF cosine on character trigrams. At 12M records we replaced it with **IDF-weighted key scores** (log N/df over rare words and word pairs). This keeps the IDF idea (rare shared words count most) at a cost that grows linearly, and the score is itself a matcher feature (`kscore`).
- **Beyond the tip:** frequency features (section 4.1) handle what no pairwise similarity can: telling a typo from a different business that looks the same.

**3. "Pay attention to country-specific address patterns."**
- **India:** conversion of 8 Indic scripts; learned state spellings (`tg → telangana`, `keralam → kerala`, `mh → maharashtra`, romanised `dilli → delhi`); removal of `Door No` / `H.No` / `Plot` fillers; Indian street words (`nagar`, `marg`, `sector`, `colony`, `cross`, `main`, `stage`, …); Indian legal forms (`Pvt Ltd`, `LLP`, `OPC`); long addresses handled by wide Source 1 keys.
- **US:** state codes ↔ full names (`ohio → oh`, `texas → tx`, …, learned from the data); street types and their typos (`aenue → ave`, `dirve → dr`); `CDP`/`CITY`/`Township` suffixes; `PMB` and `Unit` fillers; house-number formats (`00709`, `4514D`, `5001-5003`).
- **France (unseen in training):** French street types (`r./rue`, `av.`, `bd`, `allée`, `impasse`, `chemin`, `route`, `quai`, …), `N°` numbering, `bis/ter`, `CEDEX`, French legal forms (`SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SNC`, `S.A.R.L.`), and the inserted `(France)`.
- **All countries:** blocking is partitioned by country as an open set of labels, and no feature uses the country label.
- **Evidence:** per-country metrics are in section 5.4, and France's match rates on test are in line with the US and India.

**4. "Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than recall."**
- **What we did:**
  - The threshold is chosen by maximising the *official* metric, **macro** F0.5, on a separate tune group, not by accuracy or AUC.
  - Each Source 2/3 record is given to only one Source 1 entity, which removes a whole class of false positives.
  - Frequency features target decoys, the main source of false positives.
- **Evidence:** the trade-off table in section 5.5 shows the chosen point (P 0.9956, R 0.9443). The final model makes only 6,504 wrong merges against 1.44M correct matches.

**5. "Do not neglect singletons — correctly predicting 'no match' is worth a full 1.0 on that entity."**
- **What we did:**
  - Singletons (5.6%) are included in the macro-F0.5 average used to pick the threshold.
  - The one-match-per-record rule means a Source 1 entity gets a match only if some Source 2/3 record actually prefers it.
  - Entities with no candidates automatically get an empty list.
- **Evidence:** 97.9% of held-out singletons are correctly left empty (24,105 / 24,628, section 5.3). On test, 104,835 entities (6.1%) are predicted empty, in line with the training singleton rate.

**6. "Validate your own output format against the rules above before submitting."**
- **What we did:** every run ends with a built-in check of every rule:
  - exact headers
  - one row per Source 1 entity, with no missing or duplicate rows
  - only existing Source 2/3 IDs in the lists
  - no duplicate IDs in any list
  - every match also present in the candidate list

  The official `utils/validate_submission.py --check-ids` also **passes** on both files (1,732,544 rows each).

---

## 7. Other relevant information

- **Fair play:** only the provided files are used. There are no external databases, geocoding, APIs or internet data. The Indic dictionary and address synonyms are learned *only* from the training labels (the fit group). The transliteration table encodes Unicode script structure, not business data.
- **Licenses:** LightGBM (MIT) and rapidfuzz (MIT) do the modelling and string matching. numpy, pandas, scikit-learn and scipy (BSD) and pyarrow (Apache 2.0) are support libraries. There are no pretrained models or LLMs.
- **Scalability:**
  - Normalisation runs in parallel across 15 processes.
  - The blocking join is a sorted-array hash join with bounded block sizes (linear cost), run in 8 threads.
  - Similarity features use rapidfuzz's multithreaded C++ kernels.
  - Runtime on 16 cores / 16 GB: training 19 min, test inference 15 min.
- **Reproducibility:** everything is controlled by one seed (42). The trained model is saved to `model/resolver.pkl`, and `--mode predict` reproduces the test outputs without retraining.
- **Limitations and next steps:**
  1. Blocking loses 3.45% of true pairs, mostly records with no address and a generic name.
  2. India trails the US by about 1.3 F0.5 points.
  3. France has no labels, so its quality can only be sanity-checked.

  Promising extensions: a second-round model that uses each Source 2/3 record's competing candidates, and phonetic keys for heavily typo'd Indian names.

---

## 8. Conclusion
Measuring the noise before modelling paid off. Asymmetric key blocking keeps candidates at about 4 per entity while scaling linearly to 12M records. Frequency features were the most valuable single idea against the decoys that F0.5 penalises. Because the design is country-agnostic, it carries over to France unchanged. The final pipeline reaches **macro F0.5 = 0.9756** on held-out data with **99.55% precision**.

---

## Appendix

### A. Code Artefacts

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        final matches (1,732,544 rows)
│   └── candidate_pairs.tsv         blocking candidate set (8,173,383 pairs)
├── code/business_entity_resolution/
│   ├── src/                        all source code (8 modules)
│   ├── README.md                   how to reproduce end-to-end (data → blocking → matching → output)
│   └── requirements.txt            pinned dependencies
└── Documentation_template.md       this document
```

| Module | Role |
|---|---|
| `main.py` | entry point: `python src/main.py` (train and predict), `--mode fit`, `--mode predict` |
| `io_utils.py` | TSV reading (pyarrow), data profile, output writing, rule checks |
| `translit.py` | Indic script → Latin conversion |
| `normalize.py` | name and address normalisation; learning the dictionaries |
| `prepare.py` | parallel normalisation |
| `blocking.py` | key index, threaded join, top-K retrieval |
| `features.py` | filter, matcher and frequency features |
| `resolver.py` | fit/tune/eval protocol, models, threshold, decision rule, prediction |

### B. Additional Results

**Stage timings (reference run, 16 cores)**

| Stage | Train | Test |
|---|---|---|
| Loading and normalisation | 64 s (+29 s learning dictionaries) | 64 s |
| Blocking (step 2a) | 390 s | 371 s |
| Filter (step 2b) | 109 s | 78 s |
| Features | 245 s | 213 s |
| Matcher | 148 s (training) | 38 s (scoring) |
| **Total** (including loading, writing and validation) | **1,135 s** | **893 s** |

**Data profile**

| | Train | Test |
|---|---|---|
| Source 1 | 2,206,821 (US 1,323,633; India 883,188) | 1,732,544 (India 809,986; US 663,106; France 259,452) |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| Empty Source 2 / Source 3 addresses | 3.4% / 3.3% | 2.6% / 2.7% |

**Test output:** 5,775,491 matches (3.33 per Source 1 entity). 1,627,709 entities have at least one match and 104,835 are empty.
