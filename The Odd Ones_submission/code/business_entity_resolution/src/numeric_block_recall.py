import re
import pandas as pd

from data_loader import load_training_data


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


def numeric_tokens(text):
    """
    Extract numeric components from an address.

    Examples:
        8-2-67/1/A/3/1 -> 8, 2, 67, 1, 3, 1
        39001           -> 39001
    """

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


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

    return pairs[
        pairs["target_id"].notna()
    ]


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Normalizing addresses...")

    for df in [s1, s2, s3]:

        df["address_norm"] = normalize_text_series(
            df["business_address"]
        )

    print("Building true pairs...")

    pairs = parse_ground_truth(gt)

    # Sample for speed
    pairs = pairs.sample(
        n=min(
            200_000,
            len(pairs)
        ),
        random_state=42
    )

    print(
        "Pairs analyzed:",
        len(pairs)
    )

    s1_lookup = s1.set_index(
        "entity_id"
    )

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    target_lookup = target.set_index(
        "entity_id"
    )

    # --------------------------------------------------
    # Frequency of numeric tokens
    # --------------------------------------------------

    print(
        "Building numeric token frequencies..."
    )

    numeric_frequency = {}

    for row in target.itertuples(
        index=False
    ):

        tokens = numeric_tokens(
            row.address_norm
        )

        for token in tokens:

            key = (
                row.country,
                token
            )

            numeric_frequency[key] = (
                numeric_frequency.get(
                    key,
                    0
                ) + 1
            )

    print(
        "Unique numeric tokens:",
        len(numeric_frequency)
    )

    # --------------------------------------------------
    # Test thresholds
    # --------------------------------------------------

    CUTOFFS = [
        10,
        25,
        50,
        100,
        250,
        500,
        1000
    ]

    hits = {
        cutoff: 0
        for cutoff in CUTOFFS
    }

    total = len(pairs)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    for row in pairs.itertuples(
        index=False
    ):

        s1_row = s1_lookup.loc[
            row.s1_id
        ]

        target_row = target_lookup.loc[
            row.target_id
        ]

        country = s1_row["country"]

        s1_numbers = numeric_tokens(
            s1_row["address_norm"]
        )

        target_numbers = numeric_tokens(
            target_row["address_norm"]
        )

        shared = (
            s1_numbers &
            target_numbers
        )

        if not shared:
            continue

        frequencies = [
            numeric_frequency.get(
                (
                    country,
                    token
                ),
                999999999
            )
            for token in shared
        ]

        rarest = min(
            frequencies
        )

        for cutoff in CUTOFFS:

            if rarest <= cutoff:
                hits[cutoff] += 1

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("NUMERIC BLOCK RECALL")
    print("=" * 60)

    for cutoff in CUTOFFS:

        print()
        print(
            f"Frequency <= {cutoff}:"
        )

        print(
            f"{hits[cutoff] / total:.2%}"
        )


if __name__ == "__main__":
    main()
    