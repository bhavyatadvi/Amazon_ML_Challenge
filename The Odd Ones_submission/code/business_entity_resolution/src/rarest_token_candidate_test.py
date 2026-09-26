import re
from collections import defaultdict
import pandas as pd

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 2000

TOP_RAREST_VALUES = [1, 2, 3, 4]

FREQUENCY_CUTOFFS = [
    5000,
    10000,
    25000,
    50000
]


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "and", "for",
    "inc", "incorporated",
    "llc", "ltd", "limited",
    "corp", "corporation",
    "company", "co",
    "pvt", "private",
    "public", "plc", "llp",

    "group", "services", "service",
    "solutions", "industries",
    "industry", "enterprise",
    "enterprises",

    "international", "global",

    "health", "healthcare",
    "medical", "medicine",
    "hospital", "clinic",
    "care", "center", "centre",

    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "lane", "ln",

    "building", "block",
    "floor", "sector",
    "district", "city",
    "state",

    "india",
    "united",
    "states",
    "usa"
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
# PREPARE
# ============================================================

def prepare(df):

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

def parse_ground_truth(gt, sample_ids):

    sample_ids = set(sample_ids)

    result = defaultdict(set)

    subset = gt[
        gt["source1_entity_id"].isin(sample_ids)
    ]

    for row in subset.itertuples(index=False):

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
                ].add(entity_id)

    return result


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def build_frequency(target):

    print("Building token frequencies...")

    freq = defaultdict(int)

    for row in target.itertuples(index=False):

        tokens = (
            row.name_tokens
            |
            row.address_tokens
        )

        for token in tokens:

            freq[
                (row.country, token)
            ] += 1

    return freq


# ============================================================
# POSTING INDEX
# ============================================================

def build_indexes(target, freq):

    print("Building token index...")

    index = defaultdict(list)

    for row in target.itertuples(index=False):

        country = row.country

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

            index[key].append(
                row.entity_id
            )

    return index


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    row,
    index,
    freq,
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

    for token in tokens:

        key = (
            country,
            token
        )

        f = freq.get(
            key,
            999999999
        )

        if f <= cutoff:

            usable.append(
                (f, token)
            )

    # Rarest first
    usable.sort(
        key=lambda x: x[0]
    )

    selected = usable[
        :number_of_tokens
    ]

    candidates = set()

    for _, token in selected:

        key = (
            country,
            token
        )

        for entity_id in index.get(
            key,
            []
        ):

            candidates.add(
                entity_id
            )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Preparing data...")

    s1 = prepare(s1)
    s2 = prepare(s2)
    s3 = prepare(s3)

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

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
        f"S1 sample: {len(sample):,}"
    )

    print(
        f"Target records: {len(target):,}"
    )

    print("Building ground truth...")

    truth = parse_ground_truth(
        gt,
        sample_ids
    )

    print("Building frequencies...")

    freq = build_frequency(
        target
    )

    print(
        f"Unique token keys: "
        f"{len(freq):,}"
    )

    print("Building index...")

    index = build_indexes(
        target,
        freq
    )

    print(
        f"Index keys: "
        f"{len(index):,}"
    )

    print()
    print("=" * 80)
    print("RAREST TOKEN CANDIDATE RETRIEVAL")
    print("=" * 80)

    for cutoff in FREQUENCY_CUTOFFS:

        for n_tokens in TOP_RAREST_VALUES:

            total_true = 0
            total_found = 0

            candidate_counts = []

            for row in sample.itertuples(
                index=False
            ):

                s1_id = row.entity_id

                true_ids = truth.get(
                    s1_id,
                    set()
                )

                candidates = retrieve(
                    row,
                    index,
                    freq,
                    cutoff,
                    n_tokens
                )

                total_true += len(
                    true_ids
                )

                total_found += len(
                    true_ids &
                    candidates
                )

                candidate_counts.append(
                    len(candidates)
                )

            recall = (
                total_found /
                total_true *
                100
                if total_true
                else 0
            )

            counts = pd.Series(
                candidate_counts
            )

            print(
                f"Cutoff <= {cutoff:4d} | "
                f"Rarest tokens = {n_tokens} | "
                f"Recall = {recall:7.3f}% | "
                f"Mean = {counts.mean():8.1f} | "
                f"Median = {counts.median():8.1f} | "
                f"P95 = {counts.quantile(.95):8.1f} | "
                f"Max = {counts.max():8.0f}"
            )


if __name__ == "__main__":
    main()