# Business Entity Resolution — The Odd Ones

Reproducible pipeline for the ML Challenge 2026 Business Entity Resolution
task: given noisy business records from three independent sources, find all
Source 2 / Source 3 matches for each Source 1 entity.

## Requirements

- Python 3.9+
- Dependencies (see `requirements.txt`):
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

Install with:

```bash
pip install -r requirements.txt
```

## Expected directory layout

This code expects to sit inside the official `<team_name>_submission/` folder,
next to a sibling `student_resource/` folder containing the challenge dataset
(this is set by `config.py`, which resolves all paths relative to this file's
location — no path editing is needed if the layout below is preserved):

```
Amazon_ML_Challenge/
├── student_resource/
│   └── dataset/
│       ├── train/
│       │   ├── train_source1.tsv
│       │   ├── train_source2.tsv
│       │   ├── train_source3.tsv
│       │   └── train_ground_truth.tsv
│       └── test/
│           ├── test_source1.tsv
│           ├── test_source2.tsv
│           └── test_source3.tsv
└── the odd ones_submission/
    ├── output/                              # created automatically
    │   ├── matching_results.tsv             # final output (leaderboard)
    │   ├── candidate_pairs.tsv              # blocking output (audit)
    │   └── other/
    │       ├── training_pairs.tsv           # intermediate (training)
    │       ├── matcher.joblib               # trained model
    │       ├── stage1_reduced_candidates.tsv  # Stage 1 checkpoint (inference)
    │       └── matching_results_scored.tsv  # debug: match probabilities
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/                         # all source code (this folder)
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
```

## Pipeline overview

```
                 S1 record
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       exact       rare      token
       blocks      token     overlap
          │         │         │
          └─────────┼─────────┘
                    ↓
              CANDIDATE SET
                    ↓
           feature calculation
                    ↓
             MATCHING MODEL
                    ↓
             probability
                    ↓
              threshold
                    ↓
            final matched IDs
```

Two independent phases share this shape — **training** (fits the model once,
offline, using training data + ground truth) and **inference** (generates
`candidate_pairs.tsv` and `matching_results.tsv` for the test set using the
already-trained model). Training only needs to be run once; inference is what
actually produces the submission files.

## How to reproduce end-to-end

Run everything from `code/business_entity_resolution/src/`.

### Phase 1 — Train the matcher (produces `output/other/matcher.joblib`)

```bash
# 1. Sample positive/negative training pairs from the training set,
#    using exact-match blocking against 20,000 sampled S1 entities
#    (3 sampled negatives per positive).
python build_training_pairs.py

# 2. Build the 17-feature vectors for those pairs and train a
#    HistGradientBoostingClassifier (max_iter=300, learning_rate=0.08,
#    max_leaf_nodes=31, l2_regularization=1.0). Reports precision/recall/F1
#    on an 80/20 held-out split, and saves the trained model, its feature
#    list, and a default decision threshold to output/other/matcher.joblib.
python train_matcher.py
```

### Phase 2 — Generate predictions for the test set

```bash
# 3. Blocking: build exact-match and rare-token inverted indexes over the
#    combined S2+S3 test target pool, generate and lexically rank candidates
#    per S1 test entity, and write output/candidate_pairs.tsv.
python production_candidate_generator_v2.py

# 4. Matching: apply a cheap lexical pre-filter to reduce each S1's
#    candidates to its top 20 by score, score the survivors with the
#    trained classifier, and write output/matching_results.tsv (plus a
#    debug file with match probabilities). Stage 1's reduced-candidate
#    output is checkpointed to output/other/stage1_reduced_candidates.tsv,
#    so re-running this script after Stage 1 has already completed once
#    skips straight to Stage 2 instead of re-scanning the full candidate
#    file. Delete that checkpoint file (or set FORCE_RERUN_STAGE1 = True
#    at the top of the script) to force a fresh Stage 1 pass.
python production_matcher.py
```

### Phase 3 — Validate before submitting

From the `student_resource/` directory (per the challenge's own validator):

```bash
python3 utils/validate_submission.py \
    --matching "../the odd ones_submission/output/matching_results.tsv" \
    --candidate "../the odd ones_submission/output/candidate_pairs.tsv" \
    --test-dir dataset/test
```

## Output files

- **`output/matching_results.tsv`** — final matches; the only file scored on
  the leaderboard.
- **`output/candidate_pairs.tsv`** — the exact candidate set fed into the
  matching model, for blocking-quality audit.
- **`output/other/matching_results_scored.tsv`** — debug file: the same
  matches as above, plus each one's model-predicted probability.

## Notes on runtime behavior

`production_matcher.py` is written to run within bounded memory regardless
of dataset size:

- No per-entity data (token sets, character sets) is precomputed and held in
  memory — lookups resolve to a row index into compact numpy string arrays,
  and tokenization is computed on demand through a bounded LRU cache
  (`TOKEN_CACHE_SIZE`, default 300,000 entries).
- Stage 2 (ML scoring) processes candidates in bounded batches
  (`STAGE2_BATCH_SIZE`, default 200,000) with progress printed periodically,
  and folds "best match per S1" selection directly into the batch loop so the
  full scored candidate list never has to exist in memory at once.
- Multiprocessing for Stage 1 is available (`USE_MULTIPROCESSING`) but off by
  default, since each worker process holds its own copy of the lookup data —
  only enable it once a single-process run has completed cleanly, and start
  with a small `N_WORKERS`.

All of the above are constants near the top of `production_matcher.py` and
can be tuned for the machine you're running on.

## Other scripts in this folder

The scripts below were used during development for exploratory data analysis
and for evaluating candidate-retrieval strategies before settling on the
exact-match + rare-token blocking approach used in the production pipeline
above. They are not part of the reproduction steps for the final submission
files, but are included for transparency and audit:

- **EDA / analysis:** `eda.py`, `ground_truth_analysis.py`,
  `name_duplication_analysis.py`, `similarity_analysis.py`,
  `candidate_size_analysis.py`, `inspect_matches.py`
- **Blocking-strategy evaluation:** `blocking_recall.py`,
  `token_blocking_recall.py`, `token_frequency_recall.py`,
  `token_overlap_recall.py`, `numeric_block_recall.py`,
  `weighted_token_recall.py`, `rarest_token_recall.py`,
  `hybrid_block_recall.py`, `top_k_candidate_retrieval.py`,
  `hybrid_candidate_retrieval.py`, `rarest_token_candidate_test.py`
- **Earlier candidate generator versions (superseded by
  `production_candidate_generator_v2.py`):** `candidate_generator.py`,
  `hybrid_candidate_generator.py`, `production_candidate_generator.py`

## Fair play

This pipeline uses only the provided training/test data. No external
databases, APIs, or geocoding services are used at any stage.