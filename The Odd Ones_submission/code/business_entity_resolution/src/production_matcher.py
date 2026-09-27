import re
import gc
import heapq
from pathlib import Path

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


# ============================================================
# SETTINGS
# ============================================================

# Larger chunks = faster, but requires more RAM.
CHUNK_SIZE = 250_000

# Only these many candidates per S1 reach the ML model.
#
# 20 is a good starting point for speed.
# Increase to 30/50 if RAM/time allows.
TOP_K = 20

# Cheap filtering.
MIN_SHARED_TOKENS = 2

# Print progress every N chunks.
PROGRESS_EVERY = 4


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def tokenize(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_text)
    )

    df["name_tokens"] = (
        df["name_norm"]
        .map(tokenize)
    )

    df["address_tokens"] = (
        df["address_norm"]
        .map(tokenize)
    )

    return df


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("Loading TEST data...")

    s1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t"
    )

    s2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t"
    )

    s3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t"
    )

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")

    return s1, s2, s3


# ============================================================
# BUILD LOOKUPS
# ============================================================

def build_lookup(df):

    return df.set_index(
        "entity_id"
    )[
        [
            "country",
            "name_norm",
            "address_norm",
            "name_tokens",
            "address_tokens"
        ]
    ]


# ============================================================
# CANDIDATE COLUMN DETECTION
# ============================================================

def identify_columns(columns):

    s1_candidates = [
        "source1_entity_id",
        "s1_id",
        "source1_id"
    ]

    target_candidates = [
        "candidate_entity_id",
        "target_id",
        "matched_entity_id",
        "target_entity_id"
    ]

    s1_col = next(
        (
            x
            for x in s1_candidates
            if x in columns
        ),
        None
    )

    target_col = next(
        (
            x
            for x in target_candidates
            if x in columns
        ),
        None
    )

    if s1_col is None:
        raise RuntimeError(
            "Could not identify S1 column. "
            f"Columns: {list(columns)}"
        )

    if target_col is None:
        raise RuntimeError(
            "Could not identify candidate column. "
            f"Columns: {list(columns)}"
        )

    return s1_col, target_col


# ============================================================
# CHEAP SCORE
# ============================================================

def cheap_score(
    s1_row,
    target_row
):

    name1 = s1_row["name_norm"]
    name2 = target_row["name_norm"]

    address1 = s1_row["address_norm"]
    address2 = target_row["address_norm"]

    name_tokens1 = s1_row["name_tokens"]
    name_tokens2 = target_row["name_tokens"]

    address_tokens1 = s1_row["address_tokens"]
    address_tokens2 = target_row["address_tokens"]

    name_shared = (
        name_tokens1
        &
        name_tokens2
    )

    address_shared = (
        address_tokens1
        &
        address_tokens2
    )

    name_shared_count = len(
        name_shared
    )

    address_shared_count = len(
        address_shared
    )

    # Exact matches are extremely strong.
    name_exact = (
        bool(name1)
        and name1 == name2
    )

    address_exact = (
        bool(address1)
        and address1 == address2
    )

    # Normalized overlap.
    name_overlap = (
        name_shared_count
        /
        max(
            1,
            min(
                len(name_tokens1),
                len(name_tokens2)
            )
        )
    )

    address_overlap = (
        address_shared_count
        /
        max(
            1,
            min(
                len(address_tokens1),
                len(address_tokens2)
            )
        )
    )

    # Character-set similarity.
    name_chars1 = set(name1)
    name_chars2 = set(name2)

    address_chars1 = set(address1)
    address_chars2 = set(address2)

    name_union = name_chars1 | name_chars2
    address_union = (
        address_chars1
        |
        address_chars2
    )

    name_char_sim = (
        len(name_chars1 & name_chars2)
        /
        len(name_union)
        if name_union
        else 0.0
    )

    address_char_sim = (
        len(
            address_chars1
            &
            address_chars2
        )
        /
        len(address_union)
        if address_union
        else 0.0
    )

    # Cheap ranking score.
    score = 0.0

    if name_exact:
        score += 100.0

    if address_exact:
        score += 100.0

    score += (
        name_overlap * 30.0
    )

    score += (
        address_overlap * 40.0
    )

    score += (
        min(name_shared_count, 5)
        * 8.0
    )

    score += (
        min(address_shared_count, 5)
        * 10.0
    )

    score += (
        name_char_sim * 5.0
    )

    score += (
        address_char_sim * 5.0
    )

    return (
        score,
        name_shared_count,
        address_shared_count
    )


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def make_features(
    s1_row,
    target_row,
    target_is_s2,
    target_is_s3
):

    name1 = s1_row["name_norm"]
    name2 = target_row["name_norm"]

    address1 = s1_row["address_norm"]
    address2 = target_row["address_norm"]

    name_tokens1 = s1_row["name_tokens"]
    name_tokens2 = target_row["name_tokens"]

    address_tokens1 = s1_row["address_tokens"]
    address_tokens2 = target_row["address_tokens"]

    name_shared = (
        name_tokens1
        &
        name_tokens2
    )

    address_shared = (
        address_tokens1
        &
        address_tokens2
    )

    name_union = (
        name_tokens1
        |
        name_tokens2
    )

    address_union = (
        address_tokens1
        |
        address_tokens2
    )

    name_jaccard = (
        len(name_shared)
        /
        len(name_union)
        if name_union
        else 0.0
    )

    address_jaccard = (
        len(address_shared)
        /
        len(address_union)
        if address_union
        else 0.0
    )

    name_overlap = (
        len(name_shared)
        /
        max(
            1,
            min(
                len(name_tokens1),
                len(name_tokens2)
            )
        )
    )

    address_overlap = (
        len(address_shared)
        /
        max(
            1,
            min(
                len(address_tokens1),
                len(address_tokens2)
            )
        )
    )

    name_chars1 = set(name1)
    name_chars2 = set(name2)

    address_chars1 = set(address1)
    address_chars2 = set(address2)

    name_char_union = (
        name_chars1
        |
        name_chars2
    )

    address_char_union = (
        address_chars1
        |
        address_chars2
    )

    name_char_similarity = (
        len(
            name_chars1
            &
            name_chars2
        )
        /
        len(name_char_union)
        if name_char_union
        else 0.0
    )

    address_char_similarity = (
        len(
            address_chars1
            &
            address_chars2
        )
        /
        len(address_char_union)
        if address_char_union
        else 0.0
    )

    name_length1 = len(name1)
    name_length2 = len(name2)

    address_length1 = len(address1)
    address_length2 = len(address2)

    return [
        int(
            s1_row["country"]
            ==
            target_row["country"]
        ),

        int(
            bool(name1)
            and
            name1 == name2
        ),

        int(
            bool(address1)
            and
            address1 == address2
        ),

        name_jaccard,

        address_jaccard,

        name_overlap,

        address_overlap,

        len(name_shared),

        len(address_shared),

        name_char_similarity,

        address_char_similarity,

        abs(
            name_length1
            -
            name_length2
        ),

        abs(
            address_length1
            -
            address_length2
        ),

        min(
            name_length1,
            name_length2
        )
        /
        max(
            1,
            max(
                name_length1,
                name_length2
            )
        ),

        min(
            address_length1,
            address_length2
        )
        /
        max(
            1,
            max(
                address_length1,
                address_length2
            )
        ),

        int(target_is_s2),

        int(target_is_s3)
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
    "target_is_s3"
]


# ============================================================
# SCORE REDUCED CANDIDATES
# ============================================================

def score_reduced_candidates(
    candidates,
    s1_lookup,
    target_lookup,
    model
):

    features = []
    ids = []

    for s1_id, target_id in candidates:

        try:
            s1_row = s1_lookup.loc[s1_id]
            target_row = target_lookup.loc[target_id]
        except KeyError:
            continue

        if (
            s1_row["country"]
            !=
            target_row["country"]
        ):
            continue

        target_source = target_row["target_source"]

        features.append(
            make_features(
                s1_row,
                target_row,
                target_source == "S2",
                target_source == "S3"
            )
        )

        ids.append(
            (
                s1_id,
                target_id
            )
        )

    if not features:
        return []

    X = pd.DataFrame(
        features,
        columns=FEATURE_NAMES
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    return [
        (
            ids[i][0],
            ids[i][1],
            float(probabilities[i])
        )
        for i in range(len(ids))
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAST PRODUCTION MATCHER")
    print("=" * 70)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading matcher...")

    matcher = joblib.load(
        MODEL_PATH
    )

    model = matcher["model"]

    threshold = matcher.get(
        "threshold",
        0.5
    )

    print(
        f"Model: {type(model).__name__}"
    )

    print(
        f"Threshold: {threshold}"
    )

    print(
        f"Top-K per S1: {TOP_K}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    s1, s2, s3 = load_test_data()

    print()
    print("Preparing data...")

    s1 = prepare_dataframe(s1)

    s2 = prepare_dataframe(s2)

    s3 = prepare_dataframe(s3)

    # --------------------------------------------------------
    # SOURCE FLAGS
    # --------------------------------------------------------

    s2["target_source"] = "S2"
    s3["target_source"] = "S3"

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    print()
    print(
        f"Target rows: {len(target):,}"
    )

    # --------------------------------------------------------
    # LOOKUPS
    # --------------------------------------------------------

    print()
    print("Building lookups...")

    s1_lookup = build_lookup(s1)

    target_lookup = target.set_index(
        "entity_id"
    )

    # --------------------------------------------------------
    # CANDIDATE FILE
    # --------------------------------------------------------

    print()
    print("Opening candidate file...")

    if not CANDIDATE_PATH.exists():

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{CANDIDATE_PATH}"
        )

    header = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        nrows=0
    )

    s1_col, target_col = identify_columns(
        header.columns
    )

    print(
        f"S1 column: {s1_col}"
    )

    print(
        f"Target column: {target_col}"
    )

    # --------------------------------------------------------
    # BEST MATCH STORAGE
    # --------------------------------------------------------

    # Heap:
    #
    # s1_id -> [(cheap_score, target_id), ...]
    #
    # Only TOP_K candidates survive.
    top_candidates = {}

    processed = 0
    chunk_number = 0

    print()
    print(
        f"Reading candidates in chunks of "
        f"{CHUNK_SIZE:,}"
    )

    reader = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        usecols=[
            s1_col,
            target_col
        ],
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    print()
    print("STAGE 1: Cheap candidate filtering")

    for chunk in reader:

        chunk_number += 1

        for row in chunk.itertuples(
            index=False
        ):

            s1_id = getattr(
                row,
                s1_col
            )

            target_id = getattr(
                row,
                target_col
            )

            try:
                s1_row = s1_lookup.loc[
                    s1_id
                ]

                target_row = target_lookup.loc[
                    target_id
                ]

            except KeyError:
                continue

            # Country is already expected to match,
            # but keep this safeguard.
            if (
                s1_row["country"]
                !=
                target_row["country"]
            ):
                continue

            (
                score,
                name_shared,
                address_shared
            ) = cheap_score(
                s1_row,
                target_row
            )

            name_exact = (
                s1_row["name_norm"]
                != ""
                and
                s1_row["name_norm"]
                ==
                target_row["name_norm"]
            )

            address_exact = (
                s1_row["address_norm"]
                != ""
                and
                s1_row["address_norm"]
                ==
                target_row["address_norm"]
            )

            # ------------------------------------------------
            # Cheap rejection
            # ------------------------------------------------
            #
            # Keep:
            #
            # 1. exact name
            # 2. exact address
            # 3. >= 2 shared tokens
            #
            # Also keep candidates with strong combined
            # character/token score.
            #
            keep = (
                name_exact
                or
                address_exact
                or
                name_shared >= MIN_SHARED_TOKENS
                or
                address_shared >= MIN_SHARED_TOKENS
                or
                score >= 30.0
            )

            if not keep:
                continue

            heap = top_candidates.get(
                s1_id
            )

            item = (
                score,
                target_id
            )

            if heap is None:

                heap = []

                heapq.heappush(
                    heap,
                    item
                )

                top_candidates[
                    s1_id
                ] = heap

            elif len(heap) < TOP_K:

                heapq.heappush(
                    heap,
                    item
                )

            elif item[0] > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    item
                )

        processed += len(chunk)

        if (
            chunk_number
            %
            PROGRESS_EVERY
            ==
            0
        ):

            total_kept = sum(
                len(x)
                for x in top_candidates.values()
            )

            print(
                f"Processed: {processed:,} | "
                f"S1 retained: {len(top_candidates):,} | "
                f"Reduced candidates: {total_kept:,}"
            )

        del chunk

        gc.collect()

    # --------------------------------------------------------
    # CONVERT TO LIST
    # --------------------------------------------------------

    reduced_candidates = []

    for s1_id, heap in top_candidates.items():

        for score, target_id in heap:

            reduced_candidates.append(
                (
                    s1_id,
                    target_id
                )
            )

    del top_candidates

    gc.collect()

    print()
    print("=" * 70)
    print("STAGE 1 COMPLETE")
    print("=" * 70)

    print(
        f"Original candidate rows: "
        f"{processed:,}"
    )

    print(
        f"Reduced candidates: "
        f"{len(reduced_candidates):,}"
    )

    if processed:

        print(
            f"Reduction: "
            f"{100 - len(reduced_candidates) / processed * 100:.2f}%"
        )

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    print()
    print(
        "STAGE 2: ML scoring"
    )

    scored = score_reduced_candidates(
        reduced_candidates,
        s1_lookup,
        target_lookup,
        model
    )

    del reduced_candidates

    gc.collect()

    print(
        f"ML-scored candidates: "
        f"{len(scored):,}"
    )

    # --------------------------------------------------------
    # BEST MATCH PER S1
    # --------------------------------------------------------

    print()
    print(
        "Selecting best match per S1..."
    )

    best = {}

    for s1_id, target_id, probability in scored:

        if probability < threshold:
            continue

        current = best.get(
            s1_id
        )

        if (
            current is None
            or
            probability > current[1]
        ):

            best[s1_id] = (
                target_id,
                probability
            )

    del scored

    gc.collect()

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "Building final output..."
    )

    final = pd.DataFrame(
        {
            "source1_entity_id":
                s1["entity_id"]
        }
    )

    final["matched_entity_ids"] = (
        final["source1_entity_id"]
        .map(
            lambda x:
                best[x][0]
                if x in best
                else ""
        )
    )

    final.to_csv(
        FINAL_PATH,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    debug_rows = []

    for s1_id, value in best.items():

        debug_rows.append(
            (
                s1_id,
                value[0],
                value[1]
            )
        )

    debug = pd.DataFrame(
        debug_rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
            "match_probability"
        ]
    )

    debug.to_csv(
        DEBUG_PATH,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    matched = (
        final["matched_entity_ids"]
        != ""
    ).sum()

    unmatched = (
        final["matched_entity_ids"]
        == ""
    ).sum()

    print()
    print("=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(
        f"Total S1 rows: "
        f"{len(final):,}"
    )

    print(
        f"Matched rows: "
        f"{matched:,}"
    )

    print(
        f"Unmatched rows: "
        f"{unmatched:,}"
    )

    print()
    print(
        f"Final file:\n{FINAL_PATH}"
    )

    print()
    print(
        f"Debug file:\n{DEBUG_PATH}"
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()