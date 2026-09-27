# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** The Odd Ones
**Team Members:** Bhavya Tadvi (Team Leader), Srushti Soni, Daxa Dubey
**Submission Date:** 27th September 2026

---

## 1. Executive Summary

Our solution resolves business entities across Source 2 and Source 3 against
the Source 1 reference set using a three-stage funnel: rare-token and
exact-match blocking generates an initial candidate set, a cheap lexical
re-ranking pass bounds that set to the 20 best candidates per entity, and a
gradient-boosted classifier makes the final match decision. Each stage trades
an increasingly precise (and expensive) scoring function for an increasingly
small candidate set, which is what makes the pipeline tractable against a
target pool of ~10 million records without ever comparing every Source 1
record against every target record. In the production test run this took
336,675,974 raw candidate pairs down to 28,565,838 pairs (91.5% reduction)
before the classifier ever ran.

---

## 2. Methodology

### 2.1 Problem Analysis

Business identity records from independent sources rarely share a common
identifier, and the noise between sources falls into two broad categories:

- **Name noise:** legal-suffix inconsistency (Corp/Corporation, Pvt/Private,
  Ltd/Limited), DBA/trade names differing from registered names, punctuation
  differences (`&` vs. "and"), word-order transpositions, and typos.
- **Address noise:** abbreviation variants (Rd/Road, St/Street),
  transliteration differences, missing components (no PIN code or state),
  landmark-based references in place of a street address, municipal numbering
  format differences, and component reordering.

The country field spans US, India, and — in the test set only, absent from
training — France, so the pipeline treats country as an open string label
rather than a fixed enum anywhere in the blocking or feature code.

The practical effect of this noise on blocking is visible directly in our
retrieval experiments (Section 3): common tokens are far too frequent to be
useful blocking keys on their own (over 2.16M unique tokens appear across the
target corpus, and many recur in tens of thousands of records), which is why
rare-token selection and a frequency cutoff — rather than raw token overlap —
are the basis of our blocking strategy.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage: candidate generation,
then supervised matching)

**Core Innovation:** A three-tier candidate reduction funnel that keeps the
pipeline runnable at full production scale (1.7M+ Source 1 entities against a
~10M-record target pool):

1. **Blocking (candidate generation):** exact-match and rare-token inverted
   indexes cut the ~1.7M × ~10M space of possible pairs down to a bounded
   candidate set per S1.
2. **Cheap lexical re-ranking:** a fast token/character-overlap score narrows
   each S1's candidates further to its top 20 by score.
3. **ML classification:** a gradient-boosted classifier scores the reduced
   candidate set, and the highest-probability match above threshold is
   selected as the final result for that S1 entity.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:**

- **Country** — hard blocking condition; a Source 1 record is never compared
  against a target record from a different country.
- **Exact normalized name** and **exact normalized address** — business name
  and address are lowercased and stripped to alphanumeric tokens
  (`[^a-z0-9]+` → single space); any target sharing an exact normalized value
  with an S1 record is added as a candidate and is protected from the
  per-entity candidate cap below, so a clean exact match is never dropped for
  space reasons.
- **Rare tokens** — for each S1 record, the rarest tokens in the normalized
  name and address are looked up against inverted indexes
  (`token -> [entity_id, ...]`) built over the target set. Tokens occurring
  more than **500** times in the target corpus are excluded from indexing,
  since common tokens produce candidate sets too large to be useful for
  blocking. The three rarest surviving tokens are used per field (name and
  address independently).
- All candidates from the above sources are deduplicated per S1, scored with
  a lightweight lexical score (exact-field match weighted heaviest, then
  shared-token count, then a rarity-weighted bonus per shared token), and
  capped at **500 candidates per S1**.

**Candidate pairs generated:** 336,675,974 pairs across 1,732,544 Source 1
entities in the production run (test set: S1 = 1,732,544, S2 = 4,887,273,
S3 = 5,082,316; combined target pool = 9,969,589 records).

**How true matches were not lost:**

Country filtering only excludes pairs that could never match under the
problem's own rules, so it cannot remove a true match, and exact-match
candidates are exempt from the candidate cap entirely. To validate the
rare-token component itself, we ran held-out retrieval experiments on a
2,000-entity training sample (6,924 true pairs) sweeping the token-frequency
cutoff and number of rarest tokens used:

| Frequency cutoff | Rarest tokens/field | Recall | Mean candidates | Median |
|---:|---:|---:|---:|---:|
| ≤500 | 3 | 68.226% | 204.1 | 148.0 |
| ≤2,500 | 3 | 89.024% | 1,437.4 | 1,220.5 |
| ≤5,000 | 3 | 93.010% | 2,442.3 | 2,011.0 |
| ≤10,000 | 3 | 94.974% | 3,976.5 | 2,714.5 |
| ≤25,000 | 3 | 95.869% | 6,092.4 | 3,276.5 |

The production configuration (cutoff ≤500, 3 rarest tokens/field) sits at the
low-candidate end of this curve — 68.2% recall from rare-token blocking alone,
at a mean of only ~204 candidates per S1. This isolated figure understates the
pipeline's actual recall, since it excludes the exact-match channel (matches
with identical normalized name or address are captured directly, independent
of token rarity) and this test measured name/address rare tokens combined at
a single shared cutoff rather than the production pipeline's independent
per-field selection. The cutoff was chosen deliberately conservative to keep
mean candidates/S1 low enough for the downstream stages to run in bounded
time and memory at ~10M target records — the alternative denser configurations
in the table trade a meaningfully larger candidate set for each additional
few points of recall, which is a worse trade at this scale.

We also evaluated two alternative candidate-retrieval strategies on the same
2,000-entity sample before settling on the approach above:

| Method | Recall@250 | Recall@1000 | Mean candidates |
|---|---:|---:|---:|
| Top-K token retrieval (bounded posting index, freq ≤5,000) | 85.066% | 89.211% | 3,772.6 |
| Hybrid token ranking | 83.478% | 87.868% | 3,730.8 |
| Exact-match + rare-token (production) | — | — | 204.1 |

The top-K and hybrid approaches reach higher recall but at roughly 18x the
mean candidate volume of the production configuration — a cost that compounds
badly at 1.7M+ S1 entities against a 10M target pool, both in candidate-file
size and in downstream Stage 1/Stage 2 runtime. The exact-match + rare-token
approach was chosen as the better trade-off for production scale.

---

## 4. Matching Model

The 336.7M raw candidate pairs from blocking are first passed through a
**cheap lexical filter** (country match required; kept if name or address are
an exact match, share ≥2 tokens, or clear a combined lexical score threshold)
and reduced to the **top 20 candidates per S1 by score** — 28,565,838
candidate pairs in the production run (a 91.5% reduction from the raw
blocking output). Only this reduced set is scored by the ML model below.

**Features used** (17 features per candidate pair):

- **Name features:** exact-match flag, Jaccard token similarity, token
  overlap ratio, shared-token count, character-level Jaccard similarity,
  length difference, length ratio
- **Address features:** the same set as name features (exact-match flag,
  Jaccard, overlap ratio, shared-token count, character similarity, length
  difference, length ratio)
- **Other:** country-match flag, target-source indicators (`target_is_s2`,
  `target_is_s3`)

**Model type:** `HistGradientBoostingClassifier` (scikit-learn)

**Threshold selection method:** Decision threshold of 0.5 on the classifier's
predicted match probability, with the highest-probability candidate above
threshold selected as the final match per S1 (all others discarded). Given
F_0.5's precision weighting, a production run should sweep this threshold on
a held-out validation split and pick the value maximizing macro F_0.5 rather
than defaulting to 0.5 — that sweep is the natural next step once ground
truth is available for the current candidate set.

---

## 5. Results & Error Analysis

We do not yet have a macro F_0.5 score to report here — that requires running
the trained classifier against a held-out split with known ground truth,
which has not been executed as part of this write-up. The recall figures in
Section 3 characterize the blocking stage's ceiling (the maximum recall the
matching model could possibly achieve, since it can only select among
candidates blocking actually retrieves); they are not a substitute for an
end-to-end F_0.5 measurement of the full pipeline.

Based on the noise patterns identified in Section 2.1, the expected failure
modes for this pipeline are:

- **Likely false positives (wrong merges):** businesses with generic or
  highly common names (e.g. shared franchise/chain naming) that also share
  city-level or otherwise coarse address tokens, where the lexical features
  see high overlap despite being distinct physical locations.
- **Likely false negatives (missed matches):** pairs where name or address
  diverge enough in tokenization — heavy abbreviation, transliteration, or an
  address missing most of its components — that they never survive the
  rare-token blocking stage at all, and so never reach the classifier
  regardless of how well it would have scored them.

Confirming which of these actually dominates the error profile, and by how
much, requires the held-out validation run referenced above.

---

## 6. Conclusion

The pipeline's central design decision is the blocking configuration: a
conservative frequency cutoff (≤500) and small per-field rare-token count (3)
that trades recall for a candidate volume roughly 18x smaller than the
alternative retrieval strategies we tested, which is what keeps Stage 1/Stage
2 runnable in bounded time and memory at ~10 million target records. The
three-stage funnel architecture (blocking → lexical re-ranking → ML
classification) means each stage only needs to be as precise as necessary to
narrow the set for the next, rather than any single stage having to be both
fast and highly accurate at full scale. The main open item is validating and
tuning the decision threshold against macro F_0.5 on a proper held-out split,
since the current 0.5 default was not chosen against that metric.

---

## Appendix

### A. Code Artefacts

Our complete, runnable pipeline ships in this submission's zip under
`code/business_entity_resolution/src/`. Structure and entry points:

- **`production_candidate_generator_v2.py`** — Stage 1 of the pipeline
  (blocking). Reads the test source files, builds exact-match and rare-token
  inverted indexes over the combined S2+S3 target pool, generates and
  lexically ranks candidates per S1 entity, and writes
  `output/candidate_pairs.tsv`.
- **`production_matcher.py`** — Stage 2 of the pipeline (matching). Loads
  `output/candidate_pairs.tsv` and the trained model
  (`output/other/matcher.joblib`), applies a cheap lexical pre-filter to
  reduce each S1's candidates to its top 20 by score, builds the 17-feature
  vector for each surviving pair, scores them with the classifier, and writes
  the final `output/matching_results.tsv` (plus a debug file with match
  probabilities at `output/other/matching_results_scored.tsv`). Runs within
  bounded memory regardless of dataset size — no per-entity data is
  precomputed and held in memory beyond compact id→row-index lookups, and
  string tokenization is computed on demand through a bounded cache. Stage
  1's reduced-candidate output is checkpointed to disk so a Stage 2 failure
  never requires re-running the full candidate-file pass.
- **`top_k_candidate_retrieval.py`**, **`hybrid_candidate_retrieval.py`**,
  **`rarest_token_candidate_test.py`** — offline evaluation scripts used
  during development to measure recall@K and candidate-volume trade-offs for
  the blocking strategies compared in Section 3, on a held-out sample of the
  training data.

**To reproduce end-to-end:** run `production_candidate_generator_v2.py` to
produce `output/candidate_pairs.tsv`, then run `production_matcher.py` to
produce `output/matching_results.tsv`.

### B. Additional Results

The recall@K and candidate-volume curves in Section 3 (from
`top_k_candidate_retrieval.py`, `hybrid_candidate_retrieval.py`, and
`rarest_token_candidate_test.py`) are the additional results available at
this stage. Feature importances from the trained classifier and a
precision/recall curve for threshold selection are natural additions once a
held-out validation run has been performed.