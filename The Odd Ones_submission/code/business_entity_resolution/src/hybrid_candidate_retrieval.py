import re
import math
from collections import defaultdict

import pandas as pd

from data_loader import load_training_data


SAMPLE_SIZE = 2000

MAX_TOKEN_FREQUENCY = 5000

TOP_K_VALUES = [50, 100, 250, 500, 1000]


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


def build_ground_truth(gt, sample_ids):

    sample_ids = set(sample_ids)

    gt = gt[
        gt["source1_entity_id"].isin(sample_ids)
    ]

    truth = defaultdict(set)

    for row in gt.itertuples(index=False):

        if pd.isna(row.matched_entity_ids):
            continue

        value = str(row.matched_entity_ids).strip()

        if not value:
            continue

        for entity_id in value.split(","):

            entity_id = entity_id.strip()

            if entity_id:
                truth[
                    row.source1_entity_id
                ].add(entity_id)

    return truth


def build_frequencies(target):

    frequency = defaultdict(int)

    for row in target.itertuples(index=False):

        country = row.country

        for token in row.name_tokens:

            frequency[
                (country, token)
            ] += 1

        for token in row.address_tokens:

            frequency[
                (country, token)
            ] += 1

    return frequency


def build_indexes(target, frequency):

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in target.itertuples(index=False):

        country = row.country
        entity_id = row.entity_id

        for token in row.name_tokens:

            if frequency[
                (country, token)
            ] <= MAX_TOKEN_FREQUENCY:

                name_index[
                    (country, token)
                ].append(entity_id)

        for token in row.address_tokens:

            if frequency[
                (country, token)
            ] <= MAX_TOKEN_FREQUENCY:

                address_index[
                    (country, token)
                ].append(entity_id)

    return name_index, address_index


def build_idf(frequency, target_size):

    return {
        key:
        math.log(
            (target_size + 1)
            /
            (freq + 1)
        ) + 1

        for key, freq
        in frequency.items()
    }


def retrieve(row, name_index, address_index, idf):

    country = row.country

    stats = defaultdict(
        lambda: {
            "name_count": 0,
            "address_count": 0,
            "name_idf": 0.0,
            "address_idf": 0.0
        }
    )

    # ========================================================
    # NAME
    # ========================================================

    for token in row.name_tokens:

        key = (country, token)

        posting = name_index.get(
            key,
            []
        )

        weight = idf.get(
            key,
            1.0
        )

        for entity_id in posting:

            stats[entity_id]["name_count"] += 1

            stats[entity_id]["name_idf"] += weight

    # ========================================================
    # ADDRESS
    # ========================================================

    for token in row.address_tokens:

        key = (country, token)

        posting = address_index.get(
            key,
            []
        )

        weight = idf.get(
            key,
            1.0
        )

        for entity_id in posting:

            stats[entity_id]["address_count"] += 1

            stats[entity_id]["address_idf"] += weight

    return stats


def ranking_score(stat):

    name_count = stat["name_count"]
    address_count = stat["address_count"]

    name_idf = stat["name_idf"]
    address_idf = stat["address_idf"]

    total_count = (
        name_count
        +
        address_count
    )

    # Address is deliberately weighted strongly.
    score = (
        10.0 * address_count
        +
        7.0 * name_count
        +
        1.0 * address_idf
        +
        0.8 * name_idf
        +
        2.0 * total_count
    )

    return score


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
        f"S1: {len(s1):,}"
    )

    print(
        f"Target: {len(target):,}"
    )

    sample = s1.sample(
        n=min(
            SAMPLE_SIZE,
            len(s1)
        ),
        random_state=42
    )

    truth = build_ground_truth(
        gt,
        sample["entity_id"].tolist()
    )

    print("Building token frequencies...")

    frequency = build_frequencies(
        target
    )

    print(
        f"Unique token keys: "
        f"{len(frequency):,}"
    )

    print("Building indexes...")

    name_index, address_index = build_indexes(
        target,
        frequency
    )

    print(
        f"Name index keys: "
        f"{len(name_index):,}"
    )

    print(
        f"Address index keys: "
        f"{len(address_index):,}"
    )

    print("Building IDF...")

    idf = build_idf(
        frequency,
        len(target)
    )

    hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    total_true = 0

    candidate_counts = []

    print()
    print("Testing hybrid ranking...")
    print()

    for i, row in enumerate(
        sample.itertuples(index=False),
        start=1
    ):

        s1_id = row.entity_id

        true_ids = truth.get(
            s1_id,
            set()
        )

        total_true += len(
            true_ids
        )

        stats = retrieve(
            row,
            name_index,
            address_index,
            idf
        )

        ranked = sorted(
            stats.items(),
            key=lambda item:
                ranking_score(item[1]),
            reverse=True
        )

        ranked_ids = [
            entity_id
            for entity_id, _ in ranked
        ]

        candidate_counts.append(
            len(ranked_ids)
        )

        for k in TOP_K_VALUES:

            selected = set(
                ranked_ids[:k]
            )

            hits[k] += len(
                true_ids & selected
            )

        if i % 100 == 0:

            print(
                f"Processed "
                f"{i:,}/"
                f"{len(sample):,}"
            )

    print()
    print("=" * 70)
    print("HYBRID TOKEN RANKING")
    print("=" * 70)

    print(
        f"True pairs: "
        f"{total_true:,}"
    )

    print()

    for k in TOP_K_VALUES:

        recall = (
            hits[k]
            /
            total_true
            *
            100
        )

        print(
            f"Recall @ {k:4d}: "
            f"{recall:.4f}%"
        )

    counts = pd.Series(
        candidate_counts
    )

    print()
    print(
        f"Mean candidates: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median candidates: "
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
        f"Maximum: "
        f"{counts.max():,}"
    )


if __name__ == "__main__":
    main()