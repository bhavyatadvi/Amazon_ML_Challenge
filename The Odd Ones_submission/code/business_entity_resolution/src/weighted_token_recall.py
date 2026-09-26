import re
import math
import pandas as pd
from collections import defaultdict

from data_loader import load_training_data


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


def parse_ground_truth(gt):

    gt = gt[
        gt["matched_entity_ids"].notna()
        & (
            gt["matched_entity_ids"]
            .astype(str)
            .str.strip()
            != ""
        )
    ][
        [
            "source1_entity_id",
            "matched_entity_ids"
        ]
    ].copy()

    pairs = (
        gt.assign(
            target_id=
            gt["matched_entity_ids"].str.split(",")
        )
        .explode("target_id")
    )

    pairs["target_id"] = (
        pairs["target_id"]
        .astype(str)
        .str.strip()
    )

    return pairs[
        [
            "source1_entity_id",
            "target_id"
        ]
    ]


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Normalizing...")

    for df in [s1, s2, s3]:

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

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    target_lookup = target.set_index(
        "entity_id"
    )

    s1_lookup = s1.set_index(
        "entity_id"
    )

    # ========================================================
    # TOKEN FREQUENCY
    # ========================================================

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

    # ========================================================
    # IDF
    # ========================================================

    N = len(target)

    idf = {}

    for key, freq in frequency.items():

        idf[key] = math.log(
            (N + 1) /
            (freq + 1)
        ) + 1

    print(
        f"Token keys: {len(idf):,}"
    )

    # ========================================================
    # TRUE PAIRS
    # ========================================================

    pairs = parse_ground_truth(gt)

    pairs = pairs.sample(
        n=min(200_000, len(pairs)),
        random_state=42
    )

    print(
        f"Pairs analyzed: {len(pairs):,}"
    )

    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    scores = []

    for row in pairs.itertuples(index=False):

        s1_row = s1_lookup.loc[
            row.source1_entity_id
        ]

        target_row = target_lookup.loc[
            row.target_id
        ]

        name_shared = (
            s1_row["name_tokens"]
            &
            target_row["name_tokens"]
        )

        address_shared = (
            s1_row["address_tokens"]
            &
            target_row["address_tokens"]
        )

        name_score = sum(
            idf.get(
                (
                    s1_row["country"],
                    token
                ),
                1.0
            )
            for token in name_shared
        )

        address_score = sum(
            idf.get(
                (
                    s1_row["country"],
                    token
                ),
                1.0
            )
            for token in address_shared
        )

        # Give name evidence slightly more importance
        score = (
            1.5 * name_score
            +
            address_score
        )

        scores.append(score)

    score_series = pd.Series(scores)

    print()
    print("=" * 60)
    print("WEIGHTED TOKEN SCORE")
    print("=" * 60)

    print(
        f"Mean:   {score_series.mean():.3f}"
    )

    print(
        f"Median: {score_series.median():.3f}"
    )

    print(
        f"25%:    {score_series.quantile(.25):.3f}"
    )

    print(
        f"75%:    {score_series.quantile(.75):.3f}"
    )

    print(
        f"Min:    {score_series.min():.3f}"
    )

    print(
        f"Max:    {score_series.max():.3f}"
    )


if __name__ == "__main__":
    main()