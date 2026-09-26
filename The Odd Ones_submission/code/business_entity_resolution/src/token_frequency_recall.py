import re
import pandas as pd

from data_loader import load_training_data


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

    "group",
    "center",
    "centre",
    "clinic",
    "services",
    "service",
    "health",
    "medical",
}


CUTOFFS = [
    5,
    10,
    20,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
]


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

    return {
        token
        for token in text.split()
        if (
            token not in STOPWORDS
            and len(token) >= 3
        )
    }


def parse_ground_truth(gt):

    gt = gt.copy()

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    gt["matched_ids"] = (
        gt["matched_entity_ids"]
        .apply(
            lambda x: [
                i.strip()
                for i in x.split(",")
                if i.strip()
            ]
        )
    )

    pairs = gt[
        [
            "source1_entity_id",
            "matched_ids"
        ]
    ].explode(
        "matched_ids"
    )

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
    # Create lookup tables
    # --------------------------------------------------

    s1_lookup = s1.set_index(
        "entity_id"
    )

    target = pd.concat(
        [
            s2,
            s3
        ],
        ignore_index=True
    )

    target_lookup = target.set_index(
        "entity_id"
    )

    # --------------------------------------------------
    # Attach S1 information
    # --------------------------------------------------

    pairs = pairs.join(
        s1_lookup[
            [
                "name_norm",
                "address_norm",
                "country"
            ]
        ],
        on="s1_id"
    )

    pairs = pairs.rename(
        columns={
            "name_norm": "s1_name",
            "address_norm": "s1_address",
            "country": "s1_country"
        }
    )

    # --------------------------------------------------
    # Attach target information
    # --------------------------------------------------

    pairs = pairs.join(
        target_lookup[
            [
                "name_norm",
                "address_norm",
                "country"
            ]
        ],
        on="target_id",
        rsuffix="_target"
    )

    pairs = pairs.rename(
        columns={
            "name_norm": "target_name",
            "address_norm": "target_address",
            "country": "target_country"
        }
    )

    # --------------------------------------------------
    # Country
    # --------------------------------------------------

    pairs = pairs[
        pairs["s1_country"] ==
        pairs["target_country"]
    ].copy()

    print(
        "Country-compatible true pairs:",
        len(pairs)
    )

    # --------------------------------------------------
    # Calculate target token frequencies
    #
    # We calculate frequencies separately for:
    # name tokens
    # address tokens
    # --------------------------------------------------

    print("Building target token frequencies...")

    name_frequency = {}
    address_frequency = {}

    # S2 + S3
    for row in target.itertuples(
        index=False
    ):

        name_tokens = tokenize(
            row.name_norm
        )

        address_tokens = tokenize(
            row.address_norm
        )

        for token in name_tokens:

            key = (
                row.country,
                token
            )

            name_frequency[key] = (
                name_frequency.get(
                    key,
                    0
                ) + 1
            )

        for token in address_tokens:

            key = (
                row.country,
                token
            )

            address_frequency[key] = (
                address_frequency.get(
                    key,
                    0
                ) + 1
            )

    print(
        "Unique name tokens:",
        len(name_frequency)
    )

    print(
        "Unique address tokens:",
        len(address_frequency)
    )

    # --------------------------------------------------
    # Evaluate true pairs
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("TOKEN FREQUENCY RECALL")
    print("=" * 70)

    results = []

    for cutoff in CUTOFFS:

        name_hits = 0
        address_hits = 0
        combined_hits = 0

        for row in pairs.itertuples(
            index=False
        ):

            s1_name_tokens = tokenize(
                row.s1_name
            )

            target_name_tokens = tokenize(
                row.target_name
            )

            shared_name = (
                s1_name_tokens &
                target_name_tokens
            )

            name_block = any(
                name_frequency.get(
                    (
                        row.s1_country,
                        token
                    ),
                    999999999
                ) <= cutoff
                for token in shared_name
            )

            s1_address_tokens = tokenize(
                row.s1_address
            )

            target_address_tokens = tokenize(
                row.target_address
            )

            shared_address = (
                s1_address_tokens &
                target_address_tokens
            )

            address_block = any(
                address_frequency.get(
                    (
                        row.s1_country,
                        token
                    ),
                    999999999
                ) <= cutoff
                for token in shared_address
            )

            if name_block:
                name_hits += 1

            if address_block:
                address_hits += 1

            if name_block or address_block:
                combined_hits += 1

        total = len(pairs)

        results.append(
            {
                "cutoff": cutoff,
                "name_recall":
                    name_hits / total,
                "address_recall":
                    address_hits / total,
                "combined_recall":
                    combined_hits / total
            }
        )

        print()
        print(
            f"Cutoff <= {cutoff}"
        )

        print(
            f"Name:     "
            f"{name_hits / total:.2%}"
        )

        print(
            f"Address:  "
            f"{address_hits / total:.2%}"
        )

        print(
            f"Combined: "
            f"{combined_hits / total:.2%}"
        )


if __name__ == "__main__":
    main()