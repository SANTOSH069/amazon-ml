# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
The pipeline has four stages:
1. Text normalisation built around the noise we measured in the training data, including romanisation of 8 Indic scripts and dictionaries learned from training pairs.
2. A hashed-key inverted index that returns a few Source 1 candidates for each Source 2/3 record, followed by a small learned filter.
3. A LightGBM matcher that uses string similarities plus *frequency* features, which tell look-alike decoy businesses apart from true matches that contain typos.
4. A decision rule that gives each Source 2/3 record to at most one Source 1 entity.

On Source 1 entities held out from all training and tuning, the pipeline reaches **macro F0.5 = 0.9756**. It uses **4.14 candidates per Source 1 entity** (4.72 on test), and the whole pipeline runs in linear time.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from the training data (2.2M Source 1, 10.3M Source 2/3 records, 7.64M labelled pairs):
- **Match structure:** every Source 2/3 record matches **at most one** Source 1 entity (0 exceptions), and **no match crosses countries**. Source 1 entities have 3.46 matches on average (max 11), and 5.6% are singletons. 26% of Source 2/3 records match nothing.
- **Name noise:**
  - changes in case, spacing and punctuation, and accents added to letters (`Ínc`, `Láwrence`)
  - leetspeak typos (`5ecure`, `J0aquin`, `lnc`, `C0mpany`) and letter swaps
  - duplicated words (`Vetsch, Vetsch`), shuffled word order, and swapped legal suffixes (Pvt Ltd / Private Limited / LLC / …)
  - generic words appended (Services, Center, Partners), honorifics (Shri, Smt, Mr, Dr) and junk prefixes (`***`, `>>`, `#`, `@`)
  - aliases with an invented name, where the real name comes after the marker: `Xyz dba <name>`, `f/k/a`, `formerly`, `t/a`, `aka`, `nee`
  - website or hashtag names (`emerakeystone.com`, `#danieljoaquin`), 4.6% of records
  - completely invented names at the true address
  - **Indic scripts**: 28% of Indian Source 2 names, in Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada or Malayalam.
- **Address noise:**
  - Rd/Road-type abbreviations, and state codes vs full names, both directions (`OH`/`Ohio`, `Kerala`/`KL`/`Keralam`/Malayalam script)
  - `Door No` / `H.No` / `Plot` / `Unit UNIT` prefixes and `CDP`/`CITY` suffixes
  - house-number formats (`00709`, `#4514`, `4514D`, `5001-5003`)
  - reordered and missing components; 3.3% of addresses are empty.
- **Decoys:** unmatched records that look almost like a Source 1 entity, with one letter changed in a rare word (`rooposh` vs `roopesh`) or a nearby house number. These are 79% of our false positives.
- **Test set:** it adds France. France follows the same patterns (`R.`/`Rue`, `BD.`, `N°`, `S.A.R.L.`, `(FRANCE)`), so French street types and legal forms are included in the normalisation. The country label is only ever used as an open-set partition key.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (filter, matcher) + one-to-one assignment  
**Core Innovation:**
1. Asymmetric key blocking. Source 1 emits wide key sets and each Source 2/3 record a narrow one, because each Source 2/3 record is a noisy subset of its Source 1 record.
2. Target-centric retrieval, which exploits the fact that each Source 2/3 record matches at most one Source 1 entity.
3. Frequency features. A decoy's spelling repeats across several Source 2/3 records but appears in no Source 1 record; a random typo appears once.
4. Dictionaries learned from the training pairs: Indic word → Latin word, and target address spelling → Source 1 spelling.

---

## 3. Candidate Generation (Blocking)

**Normalisation first.** Accents are folded and Indic runs romanised. A learned dictionary of 1,312 Indic words (from position-aligned training pairs) is applied first, with rule-based romanisation as the fallback. Alias markers are resolved, websites unwrapped, leetspeak fixed, abbreviations canonicalised and duplicate words removed. The "core" name drops legal forms, generic words and honorifics. Addresses get 132 learned synonyms (`texas→tx`, `tg→telangana`, `aenue→ave`, romanised `dilli→delhi`) and have their filler words removed.

**Stage 1: key blocking (inverted index).**
- **Rarity:** a token's rarity is its document frequency among Source 1 records of the same country. Tokens absent from Source 1 are skipped.
- **Key types:**
  - pairs of the rarest core-name words
  - rare single name words
  - pairs of the rarest address words
  - house number × address word
  - **name word × house number** (this separates generic names such as `sky estate` that share a city)
  - name word × address word
  - run-together website names are split into Source 1 words first (`pricetotalpacific` → `price total pacific`)
- **Asymmetric widths:** Source 1 emits wide key sets (for example, house number × every address word, and pairs among its 8 rarest address words). Source 2/3 records emit narrow sets from their own rarest tokens.
- **Block purging:** keys shared by more than 40 Source 1 records are dropped. The join cost is therefore at most (#target keys × 40), which is linear in the data size.
- **Scoring and retrieval:** keys are 64-bit hashes, and the join is a sorted-array lookup (`np.searchsorted`) run in 8 threads. A pair's score is the sum of IDF weights, log(N/df), of the keys it shares. For each Source 2/3 record we keep its top-5 Source 1 records: the best one, plus any others scoring at least 40% of the best.
- **Partitioning:** the search runs separately per country string. This is an open set, so France gets its own partition automatically.

**Stage 2: learned filter.** A 150-tree LightGBM uses the key scores and ranks, rapidfuzz `token_set_ratio`/`ratio` on the core name and address, house-number agreement, and web/script/no-address flags. The cut-off is set on the tune group to keep 99.5% of the stage-1 true pairs.

- **Candidate pairs generated (train):** 16.9M after stage 1, then **9.13M final (4.14 per Source 1 entity)**.
- **Candidate pairs generated (test):** 18.2M after stage 1, then **8.17M final (4.72 per Source 1 entity)**. The test set has 5.75 Source 2/3 records per Source 1 entity vs 4.68 in train.
- **Reduction ratio (test):** 1 − 8.17M / (1.73M × 9.97M) = 0.9999995.
- **How true matches were not lost:**
  - the asymmetric key sets
  - the name × house-number keys
  - segmentation of website names
  - learned transliteration and synonyms
  - diagnosis of misses on the training data: recall at 5 candidates per target rose from 95.0% to 97.0% over three iterations
  - a filter cut-off set by recall on a separate group
  
  The final recall ceiling is 96.55%. About half of the remaining misses are Source 2/3 records with no address whose name is shared by several Source 1 entities, which are ambiguous in principle.

---

## 4. Matching Model

**Features used (52):**
- **Name features:** rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and Jaro-Winkler on the normalised and core names. Also ratio and partial ratio with spaces removed, and "concatenation cover" (the share of Source 1 words found inside a run-together name). Plus the core word counts, and counts of extra and missing words.
- **Address features:** `ratio`, `token_set_ratio`, `token_sort_ratio` and `partial_ratio` on normalised addresses; agreement of the first house number; Jaccard similarity of the number sets; address lengths; a missing-address flag.
- **Frequency features (transductive, computed on the split's own records):** how many Source 1 records and how many Source 2/3 records share each of the Source 2/3 record's name signature (sorted core words), address signature (house number + street word) and the two combined, together with the same counts for the Source 1 record's signatures. Also the highest Source 2/3 frequency and lowest Source 1 frequency of the words the Source 2/3 record adds. The third most important feature is `extra_tok_max_tdf`: a decoy's changed word repeats across several Source 2/3 records.
- **Other:** blocking score, rank, best score, number of Source 1 records hit, and number of keys shared; Source 1 context (how many Source 2/3 records propose this Source 1 record and this pair's rank); source flag; Indic-script, website and alias flags. No feature uses the country label.

**Model type:** LightGBM (MIT license, 400 trees, 127 leaves), trained on up to 8M pairs from the fit group.  
**Threshold selection method:**
- Each Source 2/3 record is assigned to its highest-probability Source 1 candidate (the one-to-one rule seen in the training labels).
- The pair is accepted if p ≥ threshold. The threshold (0.69) is chosen to maximise macro F0.5, singletons included, on the tune group.

**Validation protocol:** training Source 1 entities are split 60/20/20 into fit/tune/eval groups:
- **fit:** learns the dictionaries and trains both models
- **tune:** sets the filter cut-off and the decision threshold
- **eval:** used only for reporting

Blocking still runs against all Source 1 entities, so the held-out records face realistic competition.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - **0.9756** on the 441,521 held-out training Source 1 entities
  - 0.9752 on the tune group
  - the first version without frequency features scored 0.9696
- **Precision / recall (micro, eval):**
  - about 0.992 precision and 0.934 recall with the first model
  - an F0.5 upper bound of 0.988 if every candidate were decided correctly
- **Common false positives (wrong merges):**
  - 79% are decoys: unmatched records that copy a Source 1 entity with one letter changed or a nearby house number (`8216 bowers ln` vs `8203 bowers ln`)
  - the rest are invented-name records at an address shared by several businesses
- **Common false negatives (missed matches):**
  - records with no address and a generic name
  - true matches whose house number was also altered (the same signal that decoys carry)
  - heavily typo'd names with partial addresses
  - about 3.4% of true pairs are lost at blocking

---

## 6. Conclusion
Studying the noise closely paid off. Asymmetric key blocking keeps the candidate set at about 4 Source 1 records per entity while scaling linearly. Frequency features were the most useful single idea against the decoys that the F0.5 metric penalises. Because the design is country-agnostic (an open-set country partition, similarity-only features, and French normalisation rules), it transfers to France, where match rates on test are in line with the US and India.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/main.py` is the entry point: `python src/main.py` fits on train and writes both output files; `--mode fit` and `--mode predict` run the two halves separately.
- The modules are `translit.py`, `normalize.py`, `prepare.py`, `blocking.py`, `features.py`, `resolver.py` and `io_utils.py`.
- `README.md` has the run instructions and `requirements.txt` pins numpy, pandas, pyarrow, scipy, scikit-learn, LightGBM and rapidfuzz.

Runtime on 16 cores / 16 GB RAM: fit 19 min, test prediction 15 min.

### B. Additional Results
| stage | train (all S1) | test |
|---|---|---|
| Source 1 / Source 2+3 records | 2.21M / 10.32M | 1.73M / 9.97M |
| stage-1 pool | 16.9M pairs, recall 0.9704 | 18.2M |
| final candidates | 9.13M (4.14/S1), recall 0.9655 | 8.17M (4.72/S1) |
| matches | eval F0.5 0.9756 | 5.78M (3.33/S1), 6.0% empty |

| blocking iteration | recall@1 | recall@5 |
|---|---|---|
| symmetric rare-token keys | 0.9229 | 0.9499 |
| asymmetric wide/narrow keys | 0.9368 | 0.9608 |
| + name×house-number keys, website-name segmentation | 0.9532 | 0.9704 |
