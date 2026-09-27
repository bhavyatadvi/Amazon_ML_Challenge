"""
FAST PRODUCTION MATCHER (memory-lean)
======================================

History, so the trade-offs here make sense:

  v1 (original): pandas `.loc[single_id]` lookups inside a Python
     for-loop over the candidate file. Correct, but very slow at
     2.3 GB - each lookup pays pandas index-alignment overhead.

  v2 (first rewrite): replaced `.loc` with plain dict lookups and
     precomputed a token-frozenset AND a char-frozenset for every
     single entity, stored permanently. Fast, but at millions of
     entities this is extremely memory-hungry - a frozenset carries
     hundreds of bytes of pure Python object overhead on top of its
     contents, multiplied across millions of records, on top of the
     original DataFrames which were *also* still holding a full set
     of token-sets per row. That double bookkeeping is what crashed
     the machine (16 GB was not enough).

  v3 (this version): dict lookups are kept (still the main speed
     win over `.loc`), but NOTHING per-entity is precomputed and
     stored. Lookups resolve to a plain row index into compact numpy
     arrays (just the original strings), and tokens are computed
     on demand through a small BOUNDED cache (functools.lru_cache).
     Memory now scales with (number of entities x string length),
     not with (number of entities x frozenset overhead), and the
     cache size is a hard ceiling you control - it can never grow
     without bound no matter how big the candidate file is.

Output format, scoring formula, and keep-rules are unchanged from
the original script.
"""

import re
import gc
import heapq
from functools import lru_cache
from pathlib import Path
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd

from config import TEST_DIR, OUTPUT_DIR


# ============================================================
# PATHS
# ============================================================

OTHER_DIR = OUTPUT_DIR / "other"

MODEL_PATH = OTHER_DIR / "matcher.joblib"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

FINAL_PATH = OUTPUT_DIR / "matching_results.tsv"
DEBUG_PATH = OTHER_DIR / "matching_results_scored.tsv"

# Stage 1's output (the expensive part - one full pass over the
# candidate file) is checkpointed here. If Stage 2 crashes, errors
# out, or you need to tune STAGE2_BATCH_SIZE, the next run picks up
# from this file instead of redoing Stage 1 from scratch. Delete this
# file (or set FORCE_RERUN_STAGE1 = True) to force a fresh Stage 1.
STAGE1_CHECKPOINT_PATH = OTHER_DIR / "stage1_reduced_candidates.tsv"
FORCE_RERUN_STAGE1 = False


# ============================================================
# SETTINGS
# ============================================================

# Rows read from disk per pandas chunk.
CHUNK_SIZE = 250_000

# Rows handed to a single worker process per task (only used if
# USE_MULTIPROCESSING is True).
WORKER_BATCH_SIZE = 20_000

# Only these many candidates per S1 reach the ML model.
TOP_K = 20

# Cheap filtering.
MIN_SHARED_TOKENS = 2

# Print progress every N pandas chunks read from disk.
PROGRESS_EVERY = 4

# Tokenization results are cached so repeated S1/target rows (very
# common - the same S1 entity appears across many candidate rows)
# don't get re-split every time. This is a HARD CEILING: memory used
# by the cache can never exceed roughly this many entries, no matter
# how large the candidate file is. Lower it if you're still tight on
# RAM; raise it if you have RAM to spare and want more cache hits.
TOKEN_CACHE_SIZE = 300_000

# Parallelize Stage 1 (cheap filtering) across processes.
#
# OFF by default. Even in this leaner version, every worker process
# gets its own copy of the id->index dicts and numpy string arrays
# (sent via the Pool initializer), which multiplies memory by
# roughly N_WORKERS. Only turn this on after confirming a full
# single-process run completes cleanly, and start with a small
# N_WORKERS (2-4), not cores-1.
USE_MULTIPROCESSING = False

N_WORKERS = max(1, (cpu_count() or 2) - 1)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


@lru_cache(maxsize=TOKEN_CACHE_SIZE)
def tokenize_cached(text):
    """
    Bounded-memory replacement for storing a token set per entity.
    Same tokenization rule as the original script (tokens >= 3 chars),
    just computed on demand instead of precomputed and stored forever.
    """

    if not text:
        return frozenset()

    return frozenset(
        token
        for token in text.split()
        if len(token) >= 3
    )


# ============================================================
# PREPARE DATA
# ============================================================
#
# NOTE: unlike the original script, this does NOT add name_tokens /
# address_tokens columns holding a Python set per row. Only the
# normalized strings are kept - tokens are derived on demand via
# tokenize_cached(). This is the main memory fix.

def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)

    return df[["entity_id", "country", "name_norm", "address_norm"]]


# ============================================================
# FAST LOOKUP (id -> row index into plain numpy arrays)
# ============================================================
#
# No per-entity Python objects beyond the strings themselves and one
# dict entry (str -> int) per entity. Compare this to v2's dict of
# tuples-of-frozensets: here there is exactly one string object per
# field per entity, shared with nothing duplicated.

class FastLookup:

    __slots__ = ("index", "country", "name", "addr", "source")

    def __init__(self, df, source=None, source_col=None):

        self.index = {
            entity_id: i
            for i, entity_id in enumerate(df["entity_id"].to_numpy())
        }

        self.country = df["country"].to_numpy()
        self.name = df["name_norm"].to_numpy()
        self.addr = df["address_norm"].to_numpy()

        if source_col is not None:
            self.source = df[source_col].to_numpy()
        elif source is not None:
            self.source = np.full(len(df), source, dtype=object)
        else:
            self.source = None

    def __len__(self):
        return len(self.index)


# ============================================================
# CHEAP SCORE
# ============================================================

def cheap_score_fast(name1, addr1, name2, addr2):

    name_tokens1 = tokenize_cached(name1)
    name_tokens2 = tokenize_cached(name2)
    addr_tokens1 = tokenize_cached(addr1)
    addr_tokens2 = tokenize_cached(addr2)

    name_shared = name_tokens1 & name_tokens2
    addr_shared = addr_tokens1 & addr_tokens2

    name_shared_count = len(name_shared)
    addr_shared_count = len(addr_shared)

    name_exact = bool(name1) and name1 == name2
    addr_exact = bool(addr1) and addr1 == addr2

    name_overlap = name_shared_count / max(1, min(len(name_tokens1), len(name_tokens2)))
    addr_overlap = addr_shared_count / max(1, min(len(addr_tokens1), len(addr_tokens2)))

    name_chars1, name_chars2 = set(name1), set(name2)
    addr_chars1, addr_chars2 = set(addr1), set(addr2)

    name_union = name_chars1 | name_chars2
    addr_union = addr_chars1 | addr_chars2

    name_char_sim = len(name_chars1 & name_chars2) / len(name_union) if name_union else 0.0
    addr_char_sim = len(addr_chars1 & addr_chars2) / len(addr_union) if addr_union else 0.0

    score = 0.0

    if name_exact:
        score += 100.0
    if addr_exact:
        score += 100.0

    score += name_overlap * 30.0
    score += addr_overlap * 40.0
    score += min(name_shared_count, 5) * 8.0
    score += min(addr_shared_count, 5) * 10.0
    score += name_char_sim * 5.0
    score += addr_char_sim * 5.0

    return score, name_shared_count, addr_shared_count, name_exact, addr_exact


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def make_features_fast(name1, addr1, name2, addr2, target_is_s2, target_is_s3, country_match):

    name_tokens1 = tokenize_cached(name1)
    name_tokens2 = tokenize_cached(name2)
    addr_tokens1 = tokenize_cached(addr1)
    addr_tokens2 = tokenize_cached(addr2)

    name_shared = name_tokens1 & name_tokens2
    addr_shared = addr_tokens1 & addr_tokens2

    name_union = name_tokens1 | name_tokens2
    addr_union = addr_tokens1 | addr_tokens2

    name_jaccard = len(name_shared) / len(name_union) if name_union else 0.0
    addr_jaccard = len(addr_shared) / len(addr_union) if addr_union else 0.0

    name_overlap = len(name_shared) / max(1, min(len(name_tokens1), len(name_tokens2)))
    addr_overlap = len(addr_shared) / max(1, min(len(addr_tokens1), len(addr_tokens2)))

    name_chars1, name_chars2 = set(name1), set(name2)
    addr_chars1, addr_chars2 = set(addr1), set(addr2)

    name_char_union = name_chars1 | name_chars2
    addr_char_union = addr_chars1 | addr_chars2

    name_char_similarity = len(name_chars1 & name_chars2) / len(name_char_union) if name_char_union else 0.0
    addr_char_similarity = len(addr_chars1 & addr_chars2) / len(addr_char_union) if addr_char_union else 0.0

    name_len1, name_len2 = len(name1), len(name2)
    addr_len1, addr_len2 = len(addr1), len(addr2)

    return [
        int(country_match),
        int(bool(name1) and name1 == name2),
        int(bool(addr1) and addr1 == addr2),
        name_jaccard,
        addr_jaccard,
        name_overlap,
        addr_overlap,
        len(name_shared),
        len(addr_shared),
        name_char_similarity,
        addr_char_similarity,
        abs(name_len1 - name_len2),
        abs(addr_len1 - addr_len2),
        min(name_len1, name_len2) / max(1, max(name_len1, name_len2)),
        min(addr_len1, addr_len2) / max(1, max(addr_len1, addr_len2)),
        int(target_is_s2),
        int(target_is_s3),
    ]


FEATURE_NAMES = [
    "country_match",
    "name_exact",
    "address_exact",
    "name_jaccard",
    "address_jaccard",
    "name_overlap",
    "address_overlap",
    "name_shared_tokens",
    "address_shared_tokens",
    "name_char_similarity",
    "address_char_similarity",
    "name_length_diff",
    "address_length_diff",
    "name_length_ratio",
    "address_length_ratio",
    "target_is_s2",
    "target_is_s3",
]


# ============================================================
# CANDIDATE COLUMN DETECTION
# ============================================================

def identify_columns(columns):

    s1_candidates = ["source1_entity_id", "s1_id", "source1_id"]

    target_candidates = [
        "candidate_entity_id",
        "target_id",
        "matched_entity_id",
        "target_entity_id",
    ]

    s1_col = next((x for x in s1_candidates if x in columns), None)
    target_col = next((x for x in target_candidates if x in columns), None)

    if s1_col is None:
        raise RuntimeError(f"Could not identify S1 column. Columns: {list(columns)}")

    if target_col is None:
        raise RuntimeError(f"Could not identify candidate column. Columns: {list(columns)}")

    return s1_col, target_col


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("Loading TEST data...")

    s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t")
    s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t")
    s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t")

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")

    return s1, s2, s3


# ============================================================
# MULTIPROCESSING WORKERS (Stage 1) - only used if enabled
# ============================================================

_S1_LOOKUP = None
_TARGET_LOOKUP = None


def _init_worker(s1_lookup, target_lookup):
    global _S1_LOOKUP, _TARGET_LOOKUP
    _S1_LOOKUP = s1_lookup
    _TARGET_LOOKUP = target_lookup


def _stage1_batch(pairs):

    s1_lookup = _S1_LOOKUP
    target_lookup = _TARGET_LOOKUP

    out = []

    for s1_id, target_id in pairs:
        result = _score_pair(s1_lookup, s1_id, target_lookup, target_id)
        if result is not None:
            out.append(result)

    return out


def _batched(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


# ============================================================
# SHARED PAIR-SCORING LOGIC (used by both single-process and
# multiprocessing paths so there is exactly one implementation)
# ============================================================

def _score_pair(s1_lookup, s1_id, target_lookup, target_id):

    i1 = s1_lookup.index.get(s1_id)
    if i1 is None:
        return None

    i2 = target_lookup.index.get(target_id)
    if i2 is None:
        return None

    country1 = s1_lookup.country[i1]
    country2 = target_lookup.country[i2]

    if country1 != country2:
        return None

    name1, addr1 = s1_lookup.name[i1], s1_lookup.addr[i1]
    name2, addr2 = target_lookup.name[i2], target_lookup.addr[i2]

    score, name_shared, addr_shared, name_exact, addr_exact = cheap_score_fast(
        name1, addr1, name2, addr2
    )

    keep = (
        name_exact
        or addr_exact
        or name_shared >= MIN_SHARED_TOKENS
        or addr_shared >= MIN_SHARED_TOKENS
        or score >= 30.0
    )

    if not keep:
        return None

    return s1_id, target_id, score


# ============================================================
# STAGE 2: ML SCORING
# ============================================================
#
# Runs in bounded batches instead of building one giant feature list
# for all reduced candidates at once. Two reasons:
#
#   1. Visibility - with millions of reduced candidates, building one
#      huge list before the first model.predict_proba() call gives
#      you zero output for a long stretch. That's indistinguishable
#      from "hung" from the outside. Batching prints progress.
#
#   2. Memory - the "best match per S1" selection is folded in here,
#      so the full (s1_id, target_id, probability) list never has to
#      exist all at once either - only the current batch, plus the
#      much smaller `best` dict (at most one entry per S1) survive
#      past each batch.

STAGE2_BATCH_SIZE = 200_000
STAGE2_PROGRESS_EVERY_BATCHES = 5


def score_and_select_best(candidates, s1_lookup, target_lookup, model, threshold):

    best = {}

    total = len(candidates)
    processed = 0
    scored_count = 0
    batch_number = 0

    print(f"Scoring {total:,} reduced candidates in batches of {STAGE2_BATCH_SIZE:,}")

    for batch_start in range(0, total, STAGE2_BATCH_SIZE):

        batch_number += 1
        batch = candidates[batch_start:batch_start + STAGE2_BATCH_SIZE]

        features = []
        ids = []

        for s1_id, target_id in batch:

            i1 = s1_lookup.index.get(s1_id)
            if i1 is None:
                continue

            i2 = target_lookup.index.get(target_id)
            if i2 is None:
                continue

            country1 = s1_lookup.country[i1]
            country2 = target_lookup.country[i2]

            if country1 != country2:
                continue

            name1, addr1 = s1_lookup.name[i1], s1_lookup.addr[i1]
            name2, addr2 = target_lookup.name[i2], target_lookup.addr[i2]

            target_source = target_lookup.source[i2]

            features.append(
                make_features_fast(
                    name1, addr1, name2, addr2,
                    target_source == "S2",
                    target_source == "S3",
                    True,  # country already confirmed equal above
                )
            )

            ids.append((s1_id, target_id))

        if features:

            X = pd.DataFrame(features, columns=FEATURE_NAMES)
            probabilities = model.predict_proba(X)[:, 1]

            for i in range(len(ids)):

                probability = float(probabilities[i])

                if probability < threshold:
                    continue

                s1_id, target_id = ids[i]
                current = best.get(s1_id)

                if current is None or probability > current[1]:
                    best[s1_id] = (target_id, probability)

            scored_count += len(ids)

        processed += len(batch)

        if batch_number % STAGE2_PROGRESS_EVERY_BATCHES == 0 or processed >= total:
            print(
                f"Stage 2 processed: {processed:,} / {total:,} "
                f"({processed / total * 100:.1f}%) | "
                f"scored: {scored_count:,} | "
                f"S1 with a match so far: {len(best):,}"
            )

        del features, ids, batch
        gc.collect()

    return best


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAST PRODUCTION MATCHER (memory-lean)")
    print("=" * 70)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading matcher...")

    matcher = joblib.load(MODEL_PATH)
    model = matcher["model"]
    threshold = matcher.get("threshold", 0.5)

    print(f"Model: {type(model).__name__}")
    print(f"Threshold: {threshold}")
    print(f"Top-K per S1: {TOP_K}")
    print(f"Token cache size: {TOKEN_CACHE_SIZE:,}")
    print(f"Multiprocessing: {USE_MULTIPROCESSING} (workers={N_WORKERS})")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    s1, s2, s3 = load_test_data()

    print()
    print("Preparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    s2["target_source"] = "S2"
    s3["target_source"] = "S3"

    target = pd.concat([s2, s3], ignore_index=True)

    del s2, s3
    gc.collect()

    print(f"Target rows: {len(target):,}")

    # --------------------------------------------------------
    # LOOKUPS (compact - no per-entity token/char sets stored)
    # --------------------------------------------------------

    print()
    print("Building lookups...")

    s1_ids = s1["entity_id"].copy()

    s1_lookup = FastLookup(s1)
    target_lookup = FastLookup(target, source_col="target_source")

    print(f"S1 lookup entities: {len(s1_lookup):,}")
    print(f"Target lookup entities: {len(target_lookup):,}")

    del s1, target
    gc.collect()

    # --------------------------------------------------------
    # STAGE 1 (or load from checkpoint if it already ran before)
    # --------------------------------------------------------

    if STAGE1_CHECKPOINT_PATH.exists() and not FORCE_RERUN_STAGE1:

        print()
        print(f"Found Stage 1 checkpoint - skipping Stage 1 entirely:")
        print(f"{STAGE1_CHECKPOINT_PATH}")
        print("(delete this file, or set FORCE_RERUN_STAGE1 = True, to redo Stage 1)")

        checkpoint_df = pd.read_csv(STAGE1_CHECKPOINT_PATH, sep="\t", dtype=str)

        reduced_candidates = list(
            zip(
                checkpoint_df["source1_entity_id"].to_numpy(),
                checkpoint_df["candidate_entity_id"].to_numpy(),
            )
        )

        del checkpoint_df
        gc.collect()

        print(f"Loaded {len(reduced_candidates):,} reduced candidates from checkpoint")

    else:

        print()
        print("Opening candidate file...")

        if not CANDIDATE_PATH.exists():
            raise FileNotFoundError(f"Candidate file not found:\n{CANDIDATE_PATH}")

        header = pd.read_csv(CANDIDATE_PATH, sep="\t", nrows=0)
        s1_col, target_col = identify_columns(header.columns)

        print(f"S1 column: {s1_col}")
        print(f"Target column: {target_col}")

        top_candidates = {}

        processed = 0
        chunk_number = 0

        print()
        print(f"Reading candidates in chunks of {CHUNK_SIZE:,}")

        reader = pd.read_csv(
            CANDIDATE_PATH,
            sep="\t",
            usecols=[s1_col, target_col],
            dtype=str,
            chunksize=CHUNK_SIZE,
        )

        print()
        print("STAGE 1: Cheap candidate filtering")

        def _merge_batch_result(batch_result):
            for s1_id, target_id, score in batch_result:

                heap = top_candidates.get(s1_id)
                item = (score, target_id)

                if heap is None:
                    top_candidates[s1_id] = [item]
                elif len(heap) < TOP_K:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        pool = Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(s1_lookup, target_lookup),
        ) if USE_MULTIPROCESSING else None

        try:

            for chunk in reader:

                chunk_number += 1

                s1_arr = chunk[s1_col].to_numpy()
                target_arr = chunk[target_col].to_numpy()

                if pool is not None:

                    pairs = list(zip(s1_arr, target_arr))

                    for batch_result in pool.imap_unordered(
                        _stage1_batch,
                        _batched(pairs, WORKER_BATCH_SIZE),
                    ):
                        _merge_batch_result(batch_result)

                    del pairs

                else:

                    batch_result = []

                    for s1_id, target_id in zip(s1_arr, target_arr):
                        result = _score_pair(s1_lookup, s1_id, target_lookup, target_id)
                        if result is not None:
                            batch_result.append(result)

                    _merge_batch_result(batch_result)
                    del batch_result

                processed += len(chunk)

                if chunk_number % PROGRESS_EVERY == 0:

                    total_kept = sum(len(x) for x in top_candidates.values())

                    print(
                        f"Processed: {processed:,} | "
                        f"S1 retained: {len(top_candidates):,} | "
                        f"Reduced candidates: {total_kept:,} | "
                        f"Token cache: {tokenize_cached.cache_info().currsize:,}"
                    )

                del chunk, s1_arr, target_arr
                gc.collect()

        finally:

            if pool is not None:
                pool.close()
                pool.join()

        reduced_candidates = []

        for s1_id, heap in top_candidates.items():
            for score, target_id in heap:
                reduced_candidates.append((s1_id, target_id))

        del top_candidates
        gc.collect()

        print()
        print("=" * 70)
        print("STAGE 1 COMPLETE")
        print("=" * 70)

        print(f"Original candidate rows: {processed:,}")
        print(f"Reduced candidates: {len(reduced_candidates):,}")

        if processed:
            print(f"Reduction: {100 - len(reduced_candidates) / processed * 100:.2f}%")

        print()
        print(f"Checkpointing Stage 1 output to:\n{STAGE1_CHECKPOINT_PATH}")

        pd.DataFrame(
            reduced_candidates,
            columns=["source1_entity_id", "candidate_entity_id"],
        ).to_csv(STAGE1_CHECKPOINT_PATH, sep="\t", index=False)

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    print()
    print("STAGE 2: ML scoring")

    best = score_and_select_best(
        reduced_candidates,
        s1_lookup,
        target_lookup,
        model,
        threshold,
    )

    del reduced_candidates
    gc.collect()

    print(f"S1 entities with a match: {len(best):,}")

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("Building final output...")

    final = pd.DataFrame({"source1_entity_id": s1_ids})

    final["matched_entity_ids"] = final["source1_entity_id"].map(
        lambda x: best[x][0] if x in best else ""
    )

    final.to_csv(FINAL_PATH, sep="\t", index=False)

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    debug_rows = [
        (s1_id, value[0], value[1])
        for s1_id, value in best.items()
    ]

    debug = pd.DataFrame(
        debug_rows,
        columns=["source1_entity_id", "matched_entity_ids", "match_probability"],
    )

    debug.to_csv(DEBUG_PATH, sep="\t", index=False)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    matched = (final["matched_entity_ids"] != "").sum()
    unmatched = (final["matched_entity_ids"] == "").sum()

    print()
    print("=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(f"Total S1 rows: {len(final):,}")
    print(f"Matched rows: {matched:,}")
    print(f"Unmatched rows: {unmatched:,}")

    print()
    print(f"Final file:\n{FINAL_PATH}")
    print()
    print(f"Debug file:\n{DEBUG_PATH}")

    print()
    print("DONE.")


if __name__ == "__main__":
    main()