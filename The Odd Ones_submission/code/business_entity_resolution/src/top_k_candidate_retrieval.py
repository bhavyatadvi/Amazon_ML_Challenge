import re
import math
from collections import defaultdict
import pandas as pd

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

# We will test these K values.
TOP_K_VALUES = [50, 100, 250, 500, 1000]

# Only use token posting lists up to this size.
# Larger/common tokens are ignored for this retrieval experiment.
MAX_TOKEN_FREQUENCY = 5000

# Number of S1 entities used for this experiment.
SAMPLE_SIZE = 2000


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "and", "for", "inc", "incorporated",
    "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "pvt", "private", "public",
    "plc", "llp",

    "group", "services", "service", "solutions",
    "industries", "industry", "enterprise",
    "enterprises", "international", "global",

    "health", "healthcare", "medical", "medicine",
    "hospital", "clinic", "care", "center", "centre",

    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "building", "block", "floor",
    "sector", "district", "city", "state",

    "india", "united", "states", "usa"
}


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
        and token not in STOPWORDS
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
# GROUND TRUTH
# ============================================================

def build_true_matches(gt, sample_ids):

    sample_ids = set(sample_ids)

    gt_sample = gt[
        gt["source1_entity_id"].isin(sample_ids)
    ][
        [
            "source1_entity_id",
            "matched_entity_ids"
        ]
    ]

    true_matches = defaultdict(set)

    for row in gt_sample.itertuples(index=False):

        if pd.isna(row.matched_entity_ids):
            continue

        value = str(
            row.matched_entity_ids
        ).strip()

        if not value:
            continue

        for target_id in value.split(","):

            target_id = target_id.strip()

            if target_id:
                true_matches[
                    row.source1_entity_id
                ].add(target_id)

    return true_matches


# ============================================================
# BUILD TOKEN FREQUENCIES
# ============================================================

def build_token_frequencies(target):

    print("Building token frequencies...")

    frequency = defaultdict(int)

    for row in target.itertuples(index=False):

        tokens = (
            row.name_tokens
            |
            row.address_tokens
        )

        for token in tokens:

            frequency[
                (row.country, token)
            ] += 1

    return frequency


# ============================================================
# BUILD BOUNDED POSTING INDEX
# ============================================================

def build_posting_index(target, frequency):

    print(
        f"Building bounded posting index "
        f"(frequency <= {MAX_TOKEN_FREQUENCY})..."
    )

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    kept_name_tokens = 0
    kept_address_tokens = 0

    for row in target.itertuples(index=False):

        country = row.country
        entity_id = row.entity_id

        # ------------------------------
        # NAME TOKENS
        # ------------------------------

        for token in row.name_tokens:

            freq = frequency.get(
                (country, token),
                999999999
            )

            if freq <= MAX_TOKEN_FREQUENCY:

                name_index[
                    (country, token)
                ].append(entity_id)

                kept_name_tokens += 1

        # ------------------------------
        # ADDRESS TOKENS
        # ------------------------------

        for token in row.address_tokens:

            freq = frequency.get(
                (country, token),
                999999999
            )

            if freq <= MAX_TOKEN_FREQUENCY:

                address_index[
                    (country, token)
                ].append(entity_id)

                kept_address_tokens += 1

    print(
        f"Name posting keys: "
        f"{len(name_index):,}"
    )

    print(
        f"Address posting keys: "
        f"{len(address_index):,}"
    )

    print(
        f"Name postings: "
        f"{kept_name_tokens:,}"
    )

    print(
        f"Address postings: "
        f"{kept_address_tokens:,}"
    )

    return name_index, address_index


# ============================================================
# BUILD IDF
# ============================================================

def build_idf(frequency, target_size):

    idf = {}

    for key, freq in frequency.items():

        idf[key] = (
            math.log(
                (target_size + 1)
                /
                (freq + 1)
            )
            + 1
        )

    return idf


# ============================================================
# RETRIEVE + SCORE CANDIDATES
# ============================================================

def retrieve_candidates(
    row,
    name_index,
    address_index,
    idf
):

    country = row.country

    candidate_scores = defaultdict(float)

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    for token in row.name_tokens:

        key = (
            country,
            token
        )

        weight = idf.get(
            key,
            1.0
        )

        posting = name_index.get(
            key,
            []
        )

        # Name evidence gets more weight.
        score = 1.5 * weight

        for entity_id in posting:

            candidate_scores[
                entity_id
            ] += score

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    for token in row.address_tokens:

        key = (
            country,
            token
        )

        weight = idf.get(
            key,
            1.0
        )

        posting = address_index.get(
            key,
            []
        )

        score = weight

        for entity_id in posting:

            candidate_scores[
                entity_id
            ] += score

    # --------------------------------------------------------
    # EXACT NAME BONUS
    # --------------------------------------------------------

    # Exact normalized name will be handled separately
    # by checking candidate scores later.

    return candidate_scores


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Preparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    print(
        f"S1 records: {len(s1):,}"
    )

    print(
        f"Target records: {len(target):,}"
    )

    # ========================================================
    # SAMPLE S1
    # ========================================================

    sample_s1 = s1.sample(
        n=min(
            SAMPLE_SIZE,
            len(s1)
        ),
        random_state=42
    )

    sample_ids = (
        sample_s1["entity_id"]
        .tolist()
    )

    print(
        f"Testing S1 entities: "
        f"{len(sample_ids):,}"
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print("Building ground truth lookup...")

    true_matches = build_true_matches(
        gt,
        sample_ids
    )

    # ========================================================
    # TOKEN FREQUENCIES
    # ========================================================

    frequency = build_token_frequencies(
        target
    )

    print(
        f"Unique token keys: "
        f"{len(frequency):,}"
    )

    # ========================================================
    # INDEX
    # ========================================================

    name_index, address_index = (
        build_posting_index(
            target,
            frequency
        )
    )

    # ========================================================
    # IDF
    # ========================================================

    print("Building IDF...")

    idf = build_idf(
        frequency,
        len(target)
    )

    # ========================================================
    # TARGET LOOKUP
    # ========================================================

    target_lookup = target.set_index(
        "entity_id"
    )

    # ========================================================
    # TEST
    # ========================================================

    recall_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    total_true_pairs = 0

    candidate_counts = []

    processed = 0

    print()
    print("Testing retrieval...")

    for row in sample_s1.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        truth = true_matches.get(
            s1_id,
            set()
        )

        total_true_pairs += len(
            truth
        )

        # --------------------------------------------
        # Retrieve
        # --------------------------------------------

        scores = retrieve_candidates(
            row,
            name_index,
            address_index,
            idf
        )

        # --------------------------------------------
        # Add exact name/address candidates
        # --------------------------------------------

        if row.name_norm:

            for target_row in target.itertuples(
                index=False
            ):

                # DO NOT scan target here.
                # This branch intentionally left empty.
                # Exact matching will be added in
                # the next optimized version.

                break

        # --------------------------------------------
        # Rank
        # --------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        candidate_counts.append(
            len(ranked)
        )

        ranked_ids = [
            entity_id
            for entity_id, score
            in ranked
        ]

        # --------------------------------------------
        # Recall at K
        # --------------------------------------------

        for k in TOP_K_VALUES:

            top_k = set(
                ranked_ids[:k]
            )

            found = (
                truth
                &
                top_k
            )

            recall_hits[k] += len(
                found
            )

        processed += 1

        if processed % 100 == 0:

            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(sample_s1):,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("TOP-K TOKEN RETRIEVAL")
    print("=" * 70)

    print(
        f"True pairs: "
        f"{total_true_pairs:,}"
    )

    print()

    for k in TOP_K_VALUES:

        recall = (
            recall_hits[k]
            /
            total_true_pairs
            *
            100
            if total_true_pairs
            else 0
        )

        print(
            f"Recall @ {k:4d}: "
            f"{recall:.4f}%"
        )

    print()

    counts = pd.Series(
        candidate_counts
    )

    print(
        f"Mean retrieved candidates: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median retrieved candidates: "
        f"{counts.median():.2f}"
    )

    print(
        f"95th percentile: "
        f"{counts.quantile(.95):.2f}"
    )

    print(
        f"99th percentile: "
        f"{counts.quantile(.99):.2f}"
    )

    print(
        f"Maximum retrieved candidates: "
        f"{counts.max():,}"
    )


if __name__ == "__main__":
    main()