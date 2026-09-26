import re
import pandas as pd

from data_loader import load_training_data


def normalize_text_series(series):
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

    gt["matched_ids"] = gt["matched_entity_ids"].apply(
        lambda x: [
            i.strip()
            for i in x.split(",")
            if i.strip()
        ]
    )

    pairs = gt[
        ["source1_entity_id", "matched_ids"]
    ].explode("matched_ids")

    pairs = pairs.rename(
        columns={
            "source1_entity_id": "s1_id",
            "matched_ids": "target_id"
        }
    )

    pairs = pairs[
        pairs["target_id"].notna()
    ]

    return pairs


def add_normalized_columns(df):

    df = df.copy()

    df["name_norm"] = normalize_text_series(
        df["business_name"]
    )

    df["address_norm"] = normalize_text_series(
        df["business_address"]
    )

    return df


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Normalizing fields...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    print("Building ground-truth pairs...")

    true_pairs = parse_ground_truth(gt)

    print(
        "Total true pairs:",
        len(true_pairs)
    )

    # --------------------------------------------------
    # Combine S2 + S3
    # --------------------------------------------------

    target = pd.concat(
        [
            s2[
                [
                    "entity_id",
                    "name_norm",
                    "address_norm",
                    "country"
                ]
            ],
            s3[
                [
                    "entity_id",
                    "name_norm",
                    "address_norm",
                    "country"
                ]
            ]
        ],
        ignore_index=True
    )

    target = target.rename(
        columns={
            "entity_id": "target_id"
        }
    )

    # --------------------------------------------------
    # Merge true pairs with S1
    # --------------------------------------------------

    pairs = true_pairs.merge(
        s1[
            [
                "entity_id",
                "name_norm",
                "address_norm",
                "country"
            ]
        ],
        left_on="s1_id",
        right_on="entity_id",
        how="left"
    )

    pairs = pairs.drop(
        columns=["entity_id"]
    )

    # --------------------------------------------------
    # Add target information
    # --------------------------------------------------

    pairs = pairs.merge(
        target,
        on="target_id",
        how="left",
        suffixes=("_s1", "_target")
    )

    print("Calculating blocking keys...")

    # --------------------------------------------------
    # BLOCK A
    # country + exact name
    # --------------------------------------------------

    pairs["name_block"] = (
        (pairs["name_norm_s1"] != "") &
        (pairs["name_norm_s1"] ==
         pairs["name_norm_target"]) &
        (pairs["country_s1"] ==
         pairs["country_target"])
    )

    # --------------------------------------------------
    # BLOCK B
    # country + exact address
    # --------------------------------------------------

    pairs["address_block"] = (
        (pairs["address_norm_s1"] != "") &
        (pairs["address_norm_s1"] ==
         pairs["address_norm_target"]) &
        (pairs["country_s1"] ==
         pairs["country_target"])
    )

    # --------------------------------------------------
    # BLOCK C
    # name OR address
    # --------------------------------------------------

    pairs["name_or_address_block"] = (
        pairs["name_block"] |
        pairs["address_block"]
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    total = len(pairs)

    print()
    print("=" * 60)
    print("BLOCKING RECALL")
    print("=" * 60)

    print()
    print(
        "Total true pairs:",
        total
    )

    name_recall = (
        pairs["name_block"].sum()
        / total
    )

    address_recall = (
        pairs["address_block"].sum()
        / total
    )

    combined_recall = (
        pairs["name_or_address_block"].sum()
        / total
    )

    print()
    print(
        "Country + exact normalized name:"
    )

    print(
        f"{name_recall:.2%}"
    )

    print()
    print(
        "Country + exact normalized address:"
    )

    print(
        f"{address_recall:.2%}"
    )

    print()
    print(
        "Country + name OR address:"
    )

    print(
        f"{combined_recall:.2%}"
    )

    # --------------------------------------------------
    # Per-source recall
    # --------------------------------------------------

    pairs["target_source"] = (
        pairs["target_id"]
        .str[:2]
    )

    print()
    print("=" * 60)
    print("RECALL BY SOURCE")
    print("=" * 60)

    for source in ["S2", "S3"]:

        subset = pairs[
            pairs["target_source"] == source
        ]

        if len(subset) == 0:
            continue

        print()
        print(source)

        print(
            "True pairs:",
            len(subset)
        )

        print(
            "Name block:",
            f"{subset['name_block'].mean():.2%}"
        )

        print(
            "Address block:",
            f"{subset['address_block'].mean():.2%}"
        )

        print(
            "Combined:",
            f"{subset['name_or_address_block'].mean():.2%}"
        )


if __name__ == "__main__":
    main()