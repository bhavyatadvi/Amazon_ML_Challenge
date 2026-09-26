import re
import numpy as np
import pandas as pd

from data_loader import load_training_data


SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def normalize_text_series(series):
    """
    Fast vectorized text normalization.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_ground_truth(gt):

    gt = gt.copy()

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Convert each S1 -> list of IDs
    gt["matched_ids"] = gt["matched_entity_ids"].apply(
        lambda x: [
            i.strip()
            for i in x.split(",")
            if i.strip()
        ]
    )

    # Remove NONE entities
    gt = gt[gt["matched_ids"].str.len() > 0]

    # Explode into one true pair per row
    pairs = gt[
        ["source1_entity_id", "matched_ids"]
    ].explode("matched_ids")

    pairs = pairs.rename(
        columns={
            "source1_entity_id": "s1_id",
            "matched_ids": "matched_id"
        }
    )

    return pairs


def jaccard_from_strings(a, b):

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(
        a_tokens | b_tokens
    )


def main():

    print("Loading data...")

    s1, s2, s3, ground_truth = load_training_data()

    print("Creating true-pair table...")

    pairs = parse_ground_truth(ground_truth)

    print(
        "Total true pairs:",
        len(pairs)
    )

    # --------------------------------------------------
    # Sample true pairs
    # --------------------------------------------------

    if len(pairs) > SAMPLE_SIZE:

        pairs = pairs.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    print(
        "Analyzing sample:",
        len(pairs)
    )

    # --------------------------------------------------
    # Prepare S1
    # --------------------------------------------------

    s1_small = s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    s1_small["name_norm"] = normalize_text_series(
        s1_small["business_name"]
    )

    s1_small["address_norm"] = normalize_text_series(
        s1_small["business_address"]
    )

    # --------------------------------------------------
    # Prepare S2
    # --------------------------------------------------

    s2_small = s2[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    s2_small["name_norm"] = normalize_text_series(
        s2_small["business_name"]
    )

    s2_small["address_norm"] = normalize_text_series(
        s2_small["business_address"]
    )

    # --------------------------------------------------
    # Prepare S3
    # --------------------------------------------------

    s3_small = s3[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    s3_small["name_norm"] = normalize_text_series(
        s3_small["business_name"]
    )

    s3_small["address_norm"] = normalize_text_series(
        s3_small["business_address"]
    )

    # --------------------------------------------------
    # Combine S2 + S3
    # --------------------------------------------------

    target = pd.concat(
        [s2_small, s3_small],
        ignore_index=True
    )

    target = target.rename(
        columns={
            "entity_id": "matched_id"
        }
    )

    # --------------------------------------------------
    # Merge true pairs with S1
    # --------------------------------------------------

    result = pairs.merge(
        s1_small,
        left_on="s1_id",
        right_on="entity_id",
        how="left"
    )

    result = result.drop(
        columns=["entity_id"]
    )

    # --------------------------------------------------
    # Merge with S2/S3
    # --------------------------------------------------

    result = result.merge(
        target[
            [
                "matched_id",
                "name_norm",
                "address_norm",
                "country"
            ]
        ],
        on="matched_id",
        how="left",
        suffixes=("_s1", "_target")
    )

    # --------------------------------------------------
    # Exact normalized matching
    # --------------------------------------------------

    result["name_exact"] = (
        (result["name_norm_s1"] != "") &
        (result["name_norm_s1"] == result["name_norm_target"])
    )

    result["address_exact"] = (
        (result["address_norm_s1"] != "") &
        (result["address_norm_s1"] == result["address_norm_target"])
    )

    result["country_same"] = (
        result["country_s1"] ==
        result["country_target"]
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 50)
    print("TRUE MATCH SIMILARITY ANALYSIS")
    print("=" * 50)

    print()
    print("Pairs analyzed:", len(result))

    print()
    print(
        "Name exact after normalization:"
    )

    print(
        f"{result['name_exact'].mean():.2%}"
    )

    print()
    print(
        "Address exact after normalization:"
    )

    print(
        f"{result['address_exact'].mean():.2%}"
    )

    print()
    print(
        "Country agreement:"
    )

    print(
        f"{result['country_same'].mean():.2%}"
    )

    print()
    print(
        "Name exact + Address exact:"
    )

    both = (
        result["name_exact"] &
        result["address_exact"]
    ).mean()

    print(
        f"{both:.2%}"
    )

    # --------------------------------------------------
    # Jaccard sample
    # --------------------------------------------------

    print()
    print(
        "Calculating Jaccard similarity..."
    )

    # Only calculate for a smaller sample
    jaccard_sample = result.sample(
        n=min(20_000, len(result)),
        random_state=RANDOM_STATE
    )

    name_jaccard = []
    address_jaccard = []

    for _, row in jaccard_sample.iterrows():

        name_jaccard.append(
            jaccard_from_strings(
                row["name_norm_s1"],
                row["name_norm_target"]
            )
        )

        address_jaccard.append(
            jaccard_from_strings(
                row["address_norm_s1"],
                row["address_norm_target"]
            )
        )

    print()
    print("Name Jaccard:")
    print(
        pd.Series(name_jaccard).describe()
    )

    print()
    print("Address Jaccard:")
    print(
        pd.Series(address_jaccard).describe()
    )


if __name__ == "__main__":
    main()