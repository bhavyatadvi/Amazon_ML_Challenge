import re
import pandas as pd

from data_loader import load_training_data


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# Very common tokens are bad blocking keys.
# Ignore them.
STOPWORDS = {
    "the",
    "and",
    "of",
    "for",
    "in",
    "on",
    "at",
    "to",
    "a",
    "an",

    # Common business/legal terms
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "private",
    "pvt",
    "plc",

    # Common healthcare/business descriptors
    "group",
    "center",
    "centre",
    "clinic",
    "services",
    "service",
    "health",
    "medical",
}


def normalize_text_series(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


def tokenize(text):

    if not text:
        return set()

    tokens = text.split()

    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) >= 3
    }


def parse_ground_truth(gt):

    gt = gt.copy()

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    gt["matched_ids"] = gt[
        "matched_entity_ids"
    ].apply(
        lambda x: [
            i.strip()
            for i in x.split(",")
            if i.strip()
        ]
    )

    pairs = gt[
        [
            "source1_entity_id",
            "matched_ids"
        ]
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


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    print("Normalizing...")

    for df in [s1, s2, s3]:

        df["name_norm"] = normalize_text_series(
            df["business_name"]
        )

        df["address_norm"] = normalize_text_series(
            df["business_address"]
        )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    print("Building true pairs...")

    pairs = parse_ground_truth(gt)

    print(
        "Total true pairs:",
        len(pairs)
    )

    # --------------------------------------------------
    # S1 lookup
    # --------------------------------------------------

    s1_lookup = s1.set_index(
        "entity_id"
    )[
        [
            "name_norm",
            "address_norm",
            "country"
        ]
    ]

    # --------------------------------------------------
    # Target lookup
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

    target = target.set_index(
        "entity_id"
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    pairs = pairs.join(
        s1_lookup,
        on="s1_id",
        rsuffix="_s1"
    )

    pairs = pairs.join(
        target,
        on="target_id",
        rsuffix="_target"
    )

    # --------------------------------------------------
    # Token overlap
    # --------------------------------------------------

    print("Calculating token overlap...")

    name_token_overlap = []
    address_token_overlap = []

    for row in pairs[
        [
            "name_norm",
            "name_norm_target",
            "address_norm",
            "address_norm_target"
        ]
    ].itertuples(index=False):

        s1_name = tokenize(row.name_norm)
        target_name = tokenize(row.name_norm_target)

        s1_address = tokenize(row.address_norm)
        target_address = tokenize(row.address_norm_target)

        name_token_overlap.append(
            len(
                s1_name & target_name
            ) > 0
        )

        address_token_overlap.append(
            len(
                s1_address & target_address
            ) > 0
        )

    pairs["name_token_block"] = (
        name_token_overlap
    )

    pairs["address_token_block"] = (
        address_token_overlap
    )

    # --------------------------------------------------
    # Country
    # --------------------------------------------------

    pairs["country_same"] = (
        pairs["country"] ==
        pairs["country_target"]
    )

    # --------------------------------------------------
    # Apply country
    # --------------------------------------------------

    pairs["name_token_block"] &= (
        pairs["country_same"]
    )

    pairs["address_token_block"] &= (
        pairs["country_same"]
    )

    # --------------------------------------------------
    # Combined
    # --------------------------------------------------

    pairs["combined"] = (
        pairs["name_token_block"] |
        pairs["address_token_block"]
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TOKEN BLOCKING RECALL")
    print("=" * 60)

    total = len(pairs)

    print()
    print(
        "Total true pairs:",
        total
    )

    print()
    print(
        "Country + shared name token:"
    )

    print(
        f"{pairs['name_token_block'].mean():.2%}"
    )

    print()
    print(
        "Country + shared address token:"
    )

    print(
        f"{pairs['address_token_block'].mean():.2%}"
    )

    print()
    print(
        "Country + name token OR address token:"
    )

    print(
        f"{pairs['combined'].mean():.2%}"
    )

    # --------------------------------------------------
    # Per source
    # --------------------------------------------------

    pairs["source"] = (
        pairs["target_id"]
        .str[:2]
    )

    print()
    print("=" * 60)
    print("RECALL BY SOURCE")
    print("=" * 60)

    for source in ["S2", "S3"]:

        subset = pairs[
            pairs["source"] == source
        ]

        print()
        print(source)

        print(
            "True pairs:",
            len(subset)
        )

        print(
            "Name token:",
            f"{subset['name_token_block'].mean():.2%}"
        )

        print(
            "Address token:",
            f"{subset['address_token_block'].mean():.2%}"
        )

        print(
            "Combined:",
            f"{subset['combined'].mean():.2%}"
        )


if __name__ == "__main__":
    main()