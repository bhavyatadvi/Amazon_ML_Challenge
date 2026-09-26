import re
from collections import defaultdict
import pandas as pd

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 2000

# Rare-token retrieval settings
RARE_TOKEN_CUTOFFS = [5000, 10000, 25000]

RAREST_TOKEN_COUNTS = [1, 2, 3, 4]

# Hybrid fallback threshold.
#
# This is NOT based on ground truth.
# It is based only on the size of the rare-token candidate set.
#
# If rare-token retrieval gives fewer than this many candidates,
# we add candidates from the broader blocking strategy.
#
# You can change this after seeing the results.
HYBRID_MIN_CANDIDATES = 100


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    # Business/legal
    "the", "and", "for",
    "inc", "incorporated",
    "llc", "ltd", "limited",
    "corp", "corporation",
    "company", "co",
    "pvt", "private",
    "public", "plc", "llp",

    # Generic business
    "group", "services", "service",
    "solutions", "industries",
    "industry", "enterprise",
    "enterprises",
    "international", "global",

    # Healthcare / generic
    "health", "healthcare",
    "medical", "medicine",
    "hospital", "clinic",
    "care", "center", "centre",

    # Address
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "lane", "ln",
    "building", "block",
    "floor", "sector",
    "district", "city",
    "state",

    # Geographic
    "india",
    "united",
    "states",
    "usa",
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


# ============================================================
# TOKENIZATION
# ============================================================

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
# BUILD FREQUENCY
# ============================================================

def build_frequency(target_df):

    print("Building token frequencies...")

    frequency = defaultdict(int)

    for row in target_df.itertuples(index=False):

        country = row.country

        # Combine name and address tokens.
        # A token is counted once per entity.
        tokens = (
            row.name_tokens
            |
            row.address_tokens
        )

        for token in tokens:

            key = (
                country,
                token
            )

            frequency[key] += 1

    return frequency


# ============================================================
# BUILD TOKEN INDEX
# ============================================================

def build_token_index(
    target_df,
    frequency,
    max_frequency
):

    print(
        f"Building bounded token index "
        f"(frequency <= {max_frequency:,})..."
    )

    index = defaultdict(list)

    for row in target_df.itertuples(index=False):

        country = row.country
        entity_id = row.entity_id

        tokens = (
            row.name_tokens
            |
            row.address_tokens
        )

        for token in tokens:

            key = (
                country,
                token
            )

            freq = frequency.get(
                key,
                999999999
            )

            # Ignore extremely common tokens.
            if freq <= max_frequency:

                index[key].append(
                    entity_id
                )

    return index


# ============================================================
# BUILD FULL NAME/ADDRESS INDEX
# ============================================================

def build_baseline_indexes(target_df):

    print("Building baseline indexes...")

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in target_df.itertuples(index=False):

        country = row.country
        entity_id = row.entity_id

        for token in row.name_tokens:

            key = (
                country,
                token
            )

            name_index[key].append(
                entity_id
            )

        for token in row.address_tokens:

            key = (
                country,
                token
            )

            address_index[key].append(
                entity_id
            )

    print(
        f"Name index keys: "
        f"{len(name_index):,}"
    )

    print(
        f"Address index keys: "
        f"{len(address_index):,}"
    )

    return (
        name_index,
        address_index
    )


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def retrieve_baseline(
    row,
    name_index,
    address_index
):

    country = row.country

    candidates = set()

    # ------------------------------
    # NAME TOKENS
    # ------------------------------

    for token in row.name_tokens:

        key = (
            country,
            token
        )

        candidates.update(
            name_index.get(
                key,
                []
            )
        )

    # ------------------------------
    # ADDRESS TOKENS
    # ------------------------------

    for token in row.address_tokens:

        key = (
            country,
            token
        )

        candidates.update(
            address_index.get(
                key,
                []
            )
        )

    return candidates


# ============================================================
# RAREST TOKEN RETRIEVAL
# ============================================================

def retrieve_rarest(
    row,
    token_index,
    frequency,
    cutoff,
    number_of_tokens
):

    country = row.country

    tokens = (
        row.name_tokens
        |
        row.address_tokens
    )

    usable = []

    # --------------------------------------------------------
    # Find tokens whose posting list is not too large.
    # --------------------------------------------------------

    for token in tokens:

        key = (
            country,
            token
        )

        freq = frequency.get(
            key,
            999999999
        )

        if freq <= cutoff:

            usable.append(
                (
                    freq,
                    token
                )
            )

    # --------------------------------------------------------
    # Rarest first.
    # --------------------------------------------------------

    usable.sort(
        key=lambda x: x[0]
    )

    selected = usable[
        :number_of_tokens
    ]

    candidates = set()

    # --------------------------------------------------------
    # Retrieve candidates from selected rare tokens.
    # --------------------------------------------------------

    for _, token in selected:

        key = (
            country,
            token
        )

        candidates.update(
            token_index.get(
                key,
                []
            )
        )

    return candidates


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_hybrid(
    row,
    token_index,
    frequency,
    cutoff,
    number_of_tokens,
    name_index,
    address_index
):

    # --------------------------------------------------------
    # Stage 1:
    # Rarest-token retrieval.
    # --------------------------------------------------------

    candidates = retrieve_rarest(
        row=row,
        token_index=token_index,
        frequency=frequency,
        cutoff=cutoff,
        number_of_tokens=number_of_tokens
    )

    # --------------------------------------------------------
    # Stage 2:
    # If the rare-token block produces too few candidates,
    # add broader token candidates.
    #
    # IMPORTANT:
    # This decision uses ONLY candidate count.
    # It does NOT inspect ground truth.
    # --------------------------------------------------------

    if len(candidates) < HYBRID_MIN_CANDIDATES:

        baseline_candidates = retrieve_baseline(
            row=row,
            name_index=name_index,
            address_index=address_index
        )

        candidates.update(
            baseline_candidates
        )

    return candidates


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(
    gt,
    sample_ids
):

    sample_ids = set(
        sample_ids
    )

    result = defaultdict(set)

    subset = gt[
        gt["source1_entity_id"].isin(
            sample_ids
        )
    ][
        [
            "source1_entity_id",
            "matched_entity_ids"
        ]
    ]

    for row in subset.itertuples(
        index=False
    ):

        value = row.matched_entity_ids

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        for entity_id in value.split(","):

            entity_id = entity_id.strip()

            if entity_id:

                result[
                    row.source1_entity_id
                ].add(
                    entity_id
                )

    return result


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    sample,
    truth,
    retrieval_function,
    label
):

    total_true = 0
    total_found = 0

    candidate_counts = []

    print()
    print(
        f"Testing: {label}"
    )

    for i, row in enumerate(
        sample.itertuples(
            index=False
        ),
        start=1
    ):

        s1_id = row.entity_id

        true_ids = truth.get(
            s1_id,
            set()
        )

        candidates = retrieval_function(
            row
        )

        found = (
            true_ids &
            candidates
        )

        total_true += len(
            true_ids
        )

        total_found += len(
            found
        )

        candidate_counts.append(
            len(candidates)
        )

        if i % 500 == 0:

            print(
                f"Processed "
                f"{i:,}/"
                f"{len(sample):,}"
            )

    if total_true:

        recall = (
            total_found /
            total_true *
            100
        )

    else:

        recall = 0.0

    counts = pd.Series(
        candidate_counts
    )

    print()
    print(
        "-" * 70
    )

    print(
        f"{label}"
    )

    print(
        f"True matches:       "
        f"{total_true:,}"
    )

    print(
        f"Found by blocking:  "
        f"{total_found:,}"
    )

    print(
        f"Recall:             "
        f"{recall:.4f}%"
    )

    print(
        f"Mean candidates:    "
        f"{counts.mean():,.2f}"
    )

    print(
        f"Median candidates:  "
        f"{counts.median():,.2f}"
    )

    print(
        f"P95 candidates:     "
        f"{counts.quantile(.95):,.2f}"
    )

    print(
        f"P99 candidates:     "
        f"{counts.quantile(.99):,.2f}"
    )

    print(
        f"Maximum candidates: "
        f"{counts.max():,.0f}"
    )

    return {
        "method": label,
        "recall": recall,
        "mean": counts.mean(),
        "median": counts.median(),
        "p95": counts.quantile(.95),
        "p99": counts.quantile(.99),
        "max": counts.max()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    # ========================================================
    # PREPARE
    # ========================================================

    print("Preparing data...")

    s1 = prepare_dataframe(
        s1
    )

    s2 = prepare_dataframe(
        s2
    )

    s3 = prepare_dataframe(
        s3
    )

    target = pd.concat(
        [
            s2,
            s3
        ],
        ignore_index=True
    )

    print(
        f"S1 records: "
        f"{len(s1):,}"
    )

    print(
        f"Target records: "
        f"{len(target):,}"
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    sample = s1.sample(
        n=min(
            SAMPLE_SIZE,
            len(s1)
        ),
        random_state=42
    )

    sample_ids = sample[
        "entity_id"
    ].tolist()

    print(
        f"S1 sample: "
        f"{len(sample):,}"
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print(
        "Building ground truth..."
    )

    truth = build_ground_truth(
        gt,
        sample_ids
    )

    entities_with_matches = sum(
        1
        for entity_id in sample_ids
        if truth.get(entity_id)
    )

    print(
        f"S1 entities with matches: "
        f"{entities_with_matches:,}"
    )

    # ========================================================
    # BASELINE INDEX
    # ========================================================

    (
        name_index,
        address_index
    ) = build_baseline_indexes(
        target
    )

    # ========================================================
    # FREQUENCY
    # ========================================================

    frequency = build_frequency(
        target
    )

    print(
        f"Unique token keys: "
        f"{len(frequency):,}"
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    results = []

    # ========================================================
    # BASELINE
    # ========================================================

    result = evaluate_method(
        sample=sample,
        truth=truth,
        retrieval_function=lambda row:
            retrieve_baseline(
                row,
                name_index,
                address_index
            ),
        label="BASELINE - ALL NAME/ADDRESS TOKENS"
    )

    results.append(
        result
    )

    # ========================================================
    # RAREST TOKEN + HYBRID EXPERIMENTS
    # ========================================================

    for cutoff in RARE_TOKEN_CUTOFFS:

        print()
        print(
            "=" * 70
        )

        print(
            f"FREQUENCY CUTOFF = "
            f"{cutoff:,}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Build bounded index once for this cutoff.
        # ----------------------------------------------------

        token_index = build_token_index(
            target_df=target,
            frequency=frequency,
            max_frequency=cutoff
        )

        print(
            f"Token index keys: "
            f"{len(token_index):,}"
        )

        # ----------------------------------------------------
        # RAREST TOKEN
        # ----------------------------------------------------

        for n_tokens in RAREST_TOKEN_COUNTS:

            label = (
                f"RAREST TOKEN | "
                f"cutoff <= {cutoff:,} | "
                f"top {n_tokens}"
            )

            result = evaluate_method(
                sample=sample,
                truth=truth,
                retrieval_function=lambda row,
                    ti=token_index,
                    c=cutoff,
                    n=n_tokens:
                    retrieve_rarest(
                        row=row,
                        token_index=ti,
                        frequency=frequency,
                        cutoff=c,
                        number_of_tokens=n
                    ),
                label=label
            )

            results.append(
                result
            )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        for n_tokens in RAREST_TOKEN_COUNTS:

            label = (
                f"HYBRID | "
                f"cutoff <= {cutoff:,} | "
                f"top {n_tokens} | "
                f"fallback < {HYBRID_MIN_CANDIDATES}"
            )

            result = evaluate_method(
                sample=sample,
                truth=truth,
                retrieval_function=lambda row,
                    ti=token_index,
                    c=cutoff,
                    n=n_tokens:
                    retrieve_hybrid(
                        row=row,
                        token_index=ti,
                        frequency=frequency,
                        cutoff=c,
                        number_of_tokens=n,
                        name_index=name_index,
                        address_index=address_index
                    ),
                label=label
            )

            results.append(
                result
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print(
        "=" * 100
    )

    print(
        "HYBRID CANDIDATE GENERATOR SUMMARY"
    )

    print(
        "=" * 100
    )

    summary = pd.DataFrame(
        results
    )

    summary = summary[
        [
            "method",
            "recall",
            "mean",
            "median",
            "p95",
            "p99",
            "max"
        ]
    ]

    print(
        summary.to_string(
            index=False,
            formatters={
                "recall":
                    lambda x:
                    f"{x:.4f}%",
                "mean":
                    lambda x:
                    f"{x:,.1f}",
                "median":
                    lambda x:
                    f"{x:,.1f}",
                "p95":
                    lambda x:
                    f"{x:,.1f}",
                "p99":
                    lambda x:
                    f"{x:,.1f}",
                "max":
                    lambda x:
                    f"{x:,.0f}",
            }
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "hybrid_candidate_results.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Results saved to: "
        f"{output_file}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()