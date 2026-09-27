# ML Challenge 2026: Business Entity Resolution Solution

| | |
|---|---|
| **Team Name** | Beyond Bias |
| **Team Members** | Vedha Shree, Venkata Santosh, Sree Venkatanadh, Venkata Thanush |
| **Submission Date** | 27 September 2026 |

## 1. Executive Summary

We treat the task as a question asked of every Source 2/3 record: which Source 1 business is this, if any? The pipeline has four parts:

1. Normalisation: Built around the noise we measured in the training data. It includes conversion of eight Indic scripts to Latin letters and two dictionaries learned from the labelled pairs.
2. Two-step blocking: A hashed-key inverted index, followed by a small learned filter. This leaves 4.14 candidates per Source 1 entity (4.72 on test) while keeping 96.6% of true matches.
3. Matching model: A LightGBM model with 52 similarity and frequency features. The frequency features are what separate look-alike decoy businesses from true matches.
4. Decision rule: Each Source 2/3 record goes to at most one Source 1 entity, using a threshold tuned for macro F0.5.

On 441,521 training Source 1 entities that were never used for training or tuning, the pipeline reaches **macro F0.5 = 0.9756**. Its running time grows linearly with the data: the 1.73 million entity test set is resolved in about 18 minutes on a 16-core machine.

## 2. Methodology

### 2.1 Problem Analysis

We profiled the full training data before designing anything: 2,206,821 Source 1 records, 10,320,219 Source 2/3 records and 7,638,365 labelled pairs.

#### Structure of the matches
| Finding | Value | What we did with it |
|---|---|---|
| Source 2/3 records that match more than one Source 1 entity | 0 of 7.64 million | Each Source 2/3 record keeps only its best Source 1 candidate |
| Matches across countries | 0 | Blocking is split by country; the country value is an open set, so France is handled automatically |
| Matches per Source 1 entity | mean 3.46, max 11 | A small candidate set is enough |
| Singletons (Source 1 entities with no match) | 123,247 (5.6%) | The threshold is tuned with singletons included |
| Source 2/3 records that match nothing | 26.0% | Many non-matches must be rejected, including deliberate decoys |

#### Noise in names (measured on 300,000 matched pairs)

- Case and punctuation: Changes in case, spacing and punctuation. Junk prefixes such as `***`, `>>`, `#` and `@`.
- Leetspeak and accents: Typos like `5ecure`, `J0aquin`, `Danie1` and `lnc`, and accents added to letters (`Ínc`).
- Word-level changes: Words repeated, shuffled or dropped. Legal suffixes changed or removed; this is the most common single change (`limited` was dropped 23,873 times, `llc` 17,353 times).
- Added words: Generic words (services, center, partners) and honorifics (shri, smt, mr, dr).
- Aliases: The real name comes after a marker such as `dba`, `f/k/a`, `formerly`, `t/a` or `aka`, for example `Arcbrixx dba Rays Office`.
- Websites and hashtags: Names like `emerakeystone.com` or `#danieljoaquin`, in 4.6% of records.
- Invented names: A completely made-up name at the true address.
- Indic scripts: 27.9% of Indian Source 2 names and 18.5% of Indian Source 3 names, across Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada and Malayalam.

#### Noise in addresses
- Abbreviations: Street types written both ways (Road/Rd, Street/St) and their typos (`aenue`, `stret`).
- States: State codes and full names used in both directions (OH and Ohio, Kerala and KL and Keralam and the Malayalam spelling).
- Filler words: Door No, H.No, Plot, Unit, PMB, and city suffixes such as CDP and CITY.
- House numbers: Varied formats: `00709`, `#4514`, `4514D`, `5001-5003`.
- Missing and reordered parts: Components dropped or reordered; 3.3% of Source 2/3 addresses are empty.

Decoys: Some unmatched records copy a Source 1 business with one letter changed in a rare word (`rooposh twin company` against `roopesh twin company`) or a nearby house number (`8216 bowers ln` against `8203 bowers ln`). With our first model, 79% of false positives were of this kind.

Test set: The test set adds France (259,452 Source 1 records), which has no training labels. French records show the same kinds of noise (`R.` for Rue, `BD.`, `N°`, `S.A.R.L.`, an inserted `(FRANCE)`), so French street types and legal forms were added to the normalisation. The country value is used only to split the search, never as a model feature.

### 2.2 Solution Strategy

**Approach Type:** Blocking (hashed-key index plus a learned filter), then a gradient-boosted classifier, then one-match-per-record assignment.

#### Core Innovations
1. Asymmetric key blocking: Source 1 records emit a wide set of keys and Source 2/3 records a narrow one. A Source 2/3 record is a noisy subset of its Source 1 record, so its keys fall inside the wider set. This is what made a small candidate set compatible with high recall.
2. Retrieval per Source 2/3 record: No record belongs to two Source 1 entities, so each Source 2/3 record keeps only its few best Source 1 matches. This bounds the candidate set.
3. Frequency features against decoys: A decoy is a separate business, so its unusual spelling repeats across several Source 2/3 records while no Source 1 record has it. A random typo in a true match appears only once. These features raised macro F0.5 from 0.9696 to 0.9755 in development.
4. Dictionaries learned from the labels: Indic words to Latin words (1,312 entries) and target address spellings to Source 1 spellings (132 entries), learned only from the fit group.

Validation protocol: Training Source 1 entities are split at random (seed 42) into three separate groups:

| Group | Share | Used for |
|---|---|---|
| fit | 60% | learning the dictionaries and training both models |
| tune | 20% | choosing the filter cutoff and the decision threshold |
| eval | 20% | nothing except reporting the score |

Normalisation and blocking always run over the whole training split, so held-out entities compete against every other record, exactly as they do on the test set.

## 3. Candidate Generation (Blocking)

Blocking has two steps. Step one is an inverted index over hashed keys, which gives each Source 2/3 record a short ranked list. Step two is a small learned filter that trims that list. The output of step two is `candidate_pairs.tsv`, the exact set of pairs that the matching model scores.

### 3.1 Normalisation used by the keys

- Indic scripts: The nine Indic Unicode blocks share the Devanagari layout, so one table converts all of them (`कृष्णा` becomes `krishna`, `ಕರ್ನಾಟಕ` becomes `karnatak`). Learned dictionary entries take priority (`प्राइवेट` becomes `private`).
- Names: The cleaning steps, in order:
  1. Accents are removed.
  2. Aliases are resolved (the name after `dba` or `f/k/a` is kept).
  3. Website names are unwrapped.
  4. Dotted abbreviations are joined (`L.L.C.` becomes `llc`).
  5. `&` becomes "and".
  6. Leetspeak is repaired.
  7. Common abbreviations are standardised (`pvt` to private, `ltd` to limited, `corp` to corporation, `sté` to societe).
  8. Honorifics and repeated words are removed.

  The core name also drops legal forms (US, Indian and French: LLC, Pvt Ltd, SARL, SAS, EURL, SCI and others) and generic words.
- Addresses: Street types are abbreviated the same way on both sides, including Indian and French ones. The 132 learned synonyms are applied, filler words are removed, and house numbers are extracted without leading zeros.

### 3.2 Step one: key index

- Rarity: A word's rarity is the number of Source 1 records in the same country that contain it. Words that no Source 1 record contains cannot produce a candidate, so they are skipped.
- Key types: Every record builds keys from its rarest words:

  | Key | Built from | Why |
  |---|---|---|
  | N | pairs of the rarest core name words | robust to word order, dropped words and suffix changes |
  | M, U | the single word of a one-word name; very rare single words | short or unique names |
  | A | pairs of the rarest address words | robust to reordered or missing address parts |
  | H | house number with an address word | exact location |
  | Y | name word with house number | separates generic names such as `sky estate` or `balaji impex` that share a city |
  | X | name word with address word | survives a typo in either field |

- Key widths: Source 1 records use up to 5 name words, 8 address words and 3 house numbers, and pair each house number with every address word. Source 2/3 records use 4 name words, 4 address words, 2 house numbers and 3 address words per number.
- Website names: Run-together names are split into Source 1 words by a small dynamic program (`pricetotalpacific` becomes `price total pacific`).
- Block purging: Keys shared by more than 40 Source 1 records are dropped. This caps the work per key, so the join grows linearly with the data instead of quadratically.
- Scoring: Keys are stored as 64-bit hashes in a sorted array. Each Source 2/3 record looks its keys up with a binary search, using 8 threads over chunks of one million records. A pair's score is the sum of the IDF weights, log(N/df), of the keys it shares.
- Top candidates: Each Source 2/3 record keeps its best Source 1 match, plus up to four more that score at least 40% of the best. On the training data this rule keeps 99.996% of the true pairs in the top-5 pool at a third of its size.

Recall improvement: Each change came from studying the true pairs that were missed:

| Version | Recall at top 1 | Recall at top 5 |
|---|---|---|
| keys from each record's own rarest words | 0.9229 | 0.9499 |
| plus wide and narrow key sets | 0.9368 | 0.9608 |
| plus name with house number keys and website name splitting | 0.9532 | 0.9704 |

### 3.3 Step two: candidate filter

- Model: LightGBM with 150 trees.
- Inputs:
  - key score, key rank, best key score and number of Source 1 records hit
  - token set and plain ratio similarity on the core name and address
  - house number agreement
  - website, script and missing-address flags
- Cutoff: A probability of at least 0.0618, set on the tune group to keep 99.5% of the step one true pairs.

### 3.4 Blocking results

| | Train (all 2.21 million Source 1) | Test (1.73 million Source 1) |
|---|---|---|
| Step one pool | 16,921,294 pairs (7.67 per Source 1), recall 0.9704 | 18,194,128 pairs (10.50 per Source 1) |
| Final candidates (`candidate_pairs.tsv`) | **9,134,550 (4.14 per Source 1)** | **8,176,824 (4.72 per Source 1)** |
| Blocking recall (true pairs among the candidates) | **0.9655** | no labels |
| Reduction ratio against all Source 1 by Source 2/3 pairs | 0.9999996 | 0.9999995 |

The test set has more Source 2/3 records per Source 1 entity (5.75 against 4.68 in training), which explains its slightly larger candidate lists.

### 3.5 How true matches were kept

- Design: Wide and narrow key sets, name with house number keys, website name splitting, and the learned transliteration and synonyms (section 3.2).
- Filter cutoff: Set by recall on a separate group, never by candidate count.
- What is still missed: 3.45% of true pairs. About half of these are Source 2/3 records with no address whose name is shared by several Source 1 entities. Any key that could find them is shared by too many records, and even a person could not tell which business is meant.

## 4. Matching Model

### 4.1 Features (52 per candidate pair)

| Group | Features |
|---|---|
| Name similarity | ratio (normalised Levenshtein-style edit distance), token set, token sort, partial ratio and Jaro-Winkler on the full and core name; ratio and partial ratio with spaces removed; `joined_name_coverage` (how much of the Source 1 core name appears inside a run-together name); word counts; number of extra and missing words |
| Address similarity | ratio, token set, token sort and partial ratio; whether the first house number matches; Jaccard overlap of the house number sets; address word counts; missing address flag |
| Frequency (decoy detection) | For the name signature (sorted core words), the address signature (house number plus the next street word) and both together: whether the two records share the signature, and how many Source 1 and Source 2/3 records carry each record's signature. Also how common the words that the Source 2/3 record adds are, among other Source 2/3 records and among Source 1 records |
| Blocking context | key score, shared keys, key rank, best key score, number of Source 1 records hit, score ratio; how many Source 2/3 records propose this Source 1 entity and this pair's rank among them |
| Flags | from Source 3, Indic script, website, alias |

- No country feature: The country label is never used, which is what lets the model carry over to France.
- Frequency features use no labels: They are counted on the split being resolved (the test inputs themselves).
- Most important features (by split count): `name_ratio`, `name_token_set`, `extra_word_target_frequency` (the decoy signal), `address_token_sort`, `address_token_set`, `key_score`, `core_jaro_winkler`, `address_partial` and `target_address_in_targets`.

### 4.2 Model

**Model type:** LightGBM gradient-boosted trees (MIT license). It is trained from scratch, not pretrained, and far below the 8 billion parameter limit.

| | Candidate filter | Matching model |
|---|---|---|
| Trees | 150 | 400 |
| Settings | learning rate 0.08, 127 leaves, at least 50 samples per leaf, 80% row and column subsampling, L2 penalty 1.0 | same |
| Training data | up to 8 million pairs from the fit group | up to 8 million candidate pairs from the fit group |

### 4.3 Threshold selection and decision rule

1. One entity per record: Each Source 2/3 record keeps only its most likely Source 1 candidate, because no record belongs to two entities in the training labels.
2. Threshold: The match is accepted if its probability is at least 0.69. This value comes from a grid search from 0.05 to 0.98 that maximises macro F0.5 on the tune group, singletons included.
3. Result per entity: A Source 1 entity's matches are the Source 2/3 records that chose it. If none did, its list stays empty.

## 5. Results and Error Analysis

All numbers below are for the final model on the eval group (441,521 Source 1 entities, 1,528,407 true pairs) unless noted otherwise.

### 5.1 Main metrics

| Metric | Value |
|---|---|
| **Macro F0.5 (official metric)** | **0.9756** |
| Precision (pair level) | 0.9955 |
| Recall (pair level) | 0.9446 |
| Blocking recall | 0.9655 |
| Matching model recall on candidates | 0.9784 |
| Source 1 entities predicted exactly right | 83.0% |
| Singletons correctly left empty | 97.9% (24,105 of 24,628) |
| Macro F0.5 on the fit group | 0.9764 |
| Macro F0.5 on the tune group | 0.9752 |

### 5.2 Confusion matrix (pair level)

| | Predicted match | Predicted no match |
|---|---|---|
| True match | TP 1,443,679 | FN 84,728 (52,803 missed by blocking, 31,925 rejected by the model) |
| Not a match | FP 6,504 | TN 345,985 (candidate pairs correctly rejected) |

Roughly 10^12 other pairs were never candidates. They are all true negatives, handled by blocking.

### 5.3 Confusion matrix (entity level: any match or none)

| | Predicted some match | Predicted empty |
|---|---|---|
| Has true matches | 414,063 | 2,830 |
| True singleton | 523 | 24,105 |

### 5.4 By country (eval group)

| Country | Source 1 entities | Macro F0.5 | Precision | Recall | Blocking recall | Candidates per Source 1 |
|---|---|---|---|---|---|---|
| US | 264,911 | 0.9807 | 0.9963 | 0.9542 | 0.9758 | 4.20 |
| India | 176,610 | 0.9680 | 0.9944 | 0.9301 | 0.9500 | 4.05 |
| France (test only) | 259,452 | no labels | | | | |

India is harder: native-script names, long free-form addresses and many generic names. France has no labels, so we checked its test output for plausibility. It averages 3.2 matches per entity with 6.1% left empty, in line with the US and India and with the 5.6% singleton rate in training.

### 5.5 Precision and recall trade-off

From a development run with the same pipeline:

| Threshold | Precision | Recall | Macro F0.5 |
|---|---|---|---|
| 0.30 | 0.9855 | 0.9556 | 0.9711 |
| 0.50 | 0.9919 | 0.9507 | 0.9748 |
| 0.60 | 0.9939 | 0.9478 | 0.9754 |
| 0.70 | 0.9956 | 0.9443 | 0.9755 |
| 0.80 | 0.9969 | 0.9396 | 0.9747 |
| 0.90 | 0.9983 | 0.9303 | 0.9718 |

The curve is flat around its best point, so small shifts in the threshold cost very little.

### 5.6 Errors

- Common false positives (wrong merges):
  - Decoys: unmatched records that copy a Source 1 business with one letter changed or a nearby house number. These were 79% of false positives before the frequency features, which removed a large share of them.
  - Records with an invented name at an address shared by several businesses.
- Common false negatives (missed matches):
  1. Records with no address and a generic name (the largest group), which are ambiguous in principle.
  2. True matches whose house number was also changed. This looks exactly like a decoy, so a precision-first threshold rejects some of them.
  3. Heavily misspelt names combined with partial addresses.
  4. Blocking misses (3.45% of true pairs), concentrated in India.

## 6. How We Applied the Tips for Success

### Invest in a strong blocking and candidate generation strategy

Blocking received the most engineering effort.

- What we did: Three rounds of studying missed true pairs led to wide and narrow key sets, name with house number keys and website name splitting (section 3.2). We also added a learned filter whose cutoff is set by recall.
- Evidence:
  - Top-5 recall rose from 0.9499 to 0.9704.
  - The final candidate set is only 4.14 per Source 1 entity, just above the 3.46 true matches per entity.
  - Block purging keeps running time linear: test blocking of 1.73 million by 9.97 million records takes about 9 minutes, with a reduction ratio of 0.9999995.

### Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine)

- Levenshtein family: Edit-distance ratio, partial ratio, token sort and token set ratios, and Jaro-Winkler, on names, core names and addresses.
- Jaccard: Used on house number sets; the token set ratios capture word overlap.
- TF-IDF:
  - Our first prototype used TF-IDF cosine on character trigrams.
  - At 12 million records we replaced it with IDF-weighted key scores (log N/df over rare words and word pairs). This keeps the core TF-IDF idea, that shared rare words count most, at linear cost.
  - The key score is also a model feature.
- Beyond the tip: The frequency features in section 4.1 handle something no pairwise similarity can: telling a typo apart from a different business that looks the same.

### Pay attention to country-specific address patterns

- India:
  - Conversion of 8 Indic scripts.
  - Learned state spellings (`tg` to telangana, `keralam` to kerala, `mh` to maharashtra, `dilli` to delhi).
  - Removal of Door No, H.No and Plot.
  - Indian street words (nagar, marg, sector, colony, cross, main, stage).
  - Indian legal forms (Pvt Ltd, LLP, OPC).
  - Wide Source 1 keys for long addresses.
- US:
  - State codes and full names (learned from the data).
  - Street types and their typos (`aenue` to ave, `dirve` to dr).
  - CDP, CITY and Township suffixes; PMB and Unit filler words.
  - House number formats.
- France (unseen in training):
  - French street types (rue, avenue, boulevard, allée, impasse, chemin, route, quai).
  - `N°` numbering, bis and ter, CEDEX.
  - French legal forms (SARL, SAS, SASU, EURL, SCI, SNC).
  - The inserted "(France)".
- All countries: The search is split by country as an open set, and the country label is never a model feature. Per-country results are in section 5.4.

### Consider the precision and recall trade-off (F0.5 favours precision)

- What we did:
  - The threshold maximises macro F0.5 itself on a separate tune group, not accuracy or AUC.
  - Each Source 2/3 record goes to at most one Source 1 entity, which removes a whole class of false positives.
  - The frequency features target decoys, the main source of false positives.
- Evidence:
  - Section 5.5 shows the full trade-off.
  - The final model makes 6,504 wrong merges against 1.44 million correct matches, a precision of 0.9955.

### Do not neglect singletons

- What we did:
  - Singletons (5.6%) are part of the macro F0.5 used to choose the threshold.
  - Because of the one-entity-per-record rule, a Source 1 entity only receives a match when some Source 2/3 record prefers it.
  - Entities without candidates automatically get an empty list.
- Evidence:
  - 97.9% of held-out singletons are correctly left empty (section 5.3).
  - On the test set 6.0% of entities are predicted empty, in line with the training singleton rate.

### Validate your own output format before submitting

- Built-in check: Every run ends by checking both files against every rule:
  - exact headers
  - one row per Source 1 entity, with no missing or duplicate rows
  - only existing Source 2/3 IDs
  - no duplicate IDs within a list
  - every match present in the candidate list
- Official validator: `utils/validate_submission.py --check-ids` also passes on both files (1,732,544 rows each).

## 7. Other Relevant Information

- Fair play:
  - Only the provided files are used: no external databases, geocoding, APIs or internet data.
  - The Indic dictionary and address synonyms are learned only from the training labels of the fit group.
  - The transliteration table encodes the structure of the Unicode scripts, not any business data.
- Licenses: LightGBM (MIT) and rapidfuzz (MIT) do the modelling and string matching. numpy, pandas, scikit-learn and scipy (BSD) and pyarrow (Apache 2.0) are support libraries. No pretrained models or language models are used.
- Scalability:
  - Normalisation runs in parallel across processes.
  - The blocking join is a sorted-array hash join with capped block sizes, run in 8 threads.
  - String similarities use rapidfuzz's multithreaded C++ code.
  - Training takes about 23 minutes and test inference about 18 minutes on 16 cores with 16 GB RAM.
- Reproducibility: One seed (42) controls the group split and the models. The trained model is saved to `model/resolver.pkl`, and `--mode predict` reproduces the test outputs without retraining.
- Limitations and next steps:
  - Blocking loses 3.45% of true pairs, mostly records with no address and a generic name.
  - India trails the US by about 1.3 F0.5 points.
  - France can only be checked for plausibility, since it has no labels.
  - Promising extensions: a second-round model that uses each record's competing candidates, and phonetic keys for heavily misspelt Indian names.

## 8. Conclusion

Measuring the noise before modelling paid off:
- Blocking: Asymmetric key blocking keeps the candidate set at about four per entity while scaling linearly to 12 million records.
- Decoys: Frequency features were the most valuable single idea against the look-alike businesses that F0.5 penalises most.
- New countries: The design never relies on the country label, so it carries over to France without changes.

The final pipeline reaches a macro F0.5 of **0.9756** on held-out data with **99.55% precision**.

## Appendix

### A. Code Artefacts

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        final matches (1,732,544 rows)
│   └── candidate_pairs.tsv         blocking candidate set (8,176,824 pairs)
├── code/business_entity_resolution/
│   ├── src/                        all source code (8 modules)
│   ├── README.md                   how to reproduce the pipeline from data to output
│   └── requirements.txt            pinned dependencies
└── Documentation_template.md       this document
```

| Module | Role |
|---|---|
| `main.py` | entry point: `python src/main.py` trains and predicts; `--mode fit` and `--mode predict` run each part |
| `io_utils.py` | TSV reading, data profile, writing the outputs, rule checks |
| `translit.py` | converts Indic scripts to Latin letters |
| `normalize.py` | name and address cleaning; learns the two dictionaries |
| `prepare.py` | runs the cleaning in parallel |
| `blocking.py` | key index, threaded join, top candidates per record |
| `features.py` | candidate filter, matching and frequency features |
| `resolver.py` | training protocol, both models, threshold, decision rule, prediction |

### B. Additional Results

#### Data profile
| | Train | Test |
|---|---|---|
| Source 1 | 2,206,821 (US 1,323,633; India 883,188) | 1,732,544 (India 809,986; US 663,106; France 259,452) |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| Empty Source 2 and Source 3 addresses | 3.4% and 3.3% | 2.6% and 2.7% |

Test output: 5,776,256 matches (3.33 per Source 1 entity). 1,627,740 entities have at least one match and 104,804 are empty.
