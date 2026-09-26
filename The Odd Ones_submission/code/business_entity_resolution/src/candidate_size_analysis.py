import re
import pandas as pd

from data_loader import load_test_data


STOPWORDS = {
    "the", "and", "of", "for", "in", "on", "at", "to",
    "a", "an",

    "limited", "ltd", "llp", "inc", "incorporated",
    "corp", "corporation", "company", "co",
    "private", "pvt", "plc",

    "group", "center", "centre", "clinic",
    "services", "service", "health", "medical",
}


def normalize_series(series):

    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def tokenize_series(series):

    return (
        series
        .str.split()
        .apply(
            lambda tokens: [
                token
                for token in tokens
                if (
                    token not in STOPWORDS
                    and len(token) >= 3
                )
            ]
        )
    )


def build_token_table(df, source_name):

    temp = df[
        [
            "entity_id",
            "country",
            "business_name",
            "business_address"
        ]
    ].copy()

    temp["name_norm"] = normalize_series(
        temp["business_name"]
    )

    temp["address_norm"] = normalize_series(
        temp["business_address"]
    )

    # Name tokens
    name = temp[
        [
            "entity_id",
            "country",
            "name_norm"
        ]
    ].copy()

    name["token"] = tokenize_series(
        name["name_norm"]
    )

    name = name.explode(
        "token"
    )

    name = name[
        name["token"].notna() &
        (name["token"] != "")
    ]

    name["block_type"] = "name"

    # Address tokens
    address = temp[
        [
            "entity_id",
            "country",
            "address_norm"
        ]
    ].copy()

    address["token"] = tokenize_series(
        address["address_norm"]
    )

    address = address.explode(
        "token"
    )

    address = address[
        address["token"].notna() &
        (address["token"] != "")
    ]

    address["block_type"] = "address"

    return pd.concat(
        [
            name[
                [
                    "entity_id",
                    "country",
                    "token",
                    "block_type"
                ]
            ],
            address[
                [
                    "entity_id",
                    "country",
                    "token",
                    "block_type"
                ]
            ]
        ],
        ignore_index=True
    )


def analyze_target(
    s1,
    target,
    target_name
):

    print()
    print("=" * 60)
    print(target_name)
    print("=" * 60)

    print("Building target token table...")

    target_tokens = build_token_table(
        target,
        target_name
    )

    print(
        "Target token rows:",
        len(target_tokens)
    )

    # --------------------------------------------------
    # Remove tokens that occur excessively often
    # --------------------------------------------------
    #
    # This is NOT part of the final blocking rule yet.
    # It prevents extremely common tokens from exploding
    # the analysis.
    #
    # We will measure them separately later.
    # --------------------------------------------------

    token_frequency = (
        target_tokens
        .groupby(
            [
                "country",
                "token"
            ]
        )["entity_id"]
        .nunique()
        .reset_index(
            name="target_count"
        )
    )

    print()
    print("Target token frequency:")
    print(
        token_frequency["target_count"].describe()
    )

    # --------------------------------------------------
    # Keep tokens that occur in <= 5000 target records
    # --------------------------------------------------

    useful_tokens = token_frequency[
        token_frequency["target_count"] <= 5000
    ][
        [
            "country",
            "token"
        ]
    ]

    target_tokens = target_tokens.merge(
        useful_tokens,
        on=[
            "country",
            "token"
        ],
        how="inner"
    )

    print()
    print(
        "Token rows after frequency filter:",
        len(target_tokens)
    )

    # --------------------------------------------------
    # S1 tokens
    # --------------------------------------------------

    s1_tokens = build_token_table(
        s1,
        "S1"
    )

    # We only need S1 token identity
    s1_tokens = s1_tokens[
        [
            "entity_id",
            "country",
            "token",
            "block_type"
        ]
    ]

    # --------------------------------------------------
    # Merge matching country + token
    # --------------------------------------------------

    print()
    print(
        "Generating token candidates..."
    )

    candidates = s1_tokens.merge(
        target_tokens[
            [
                "entity_id",
                "country",
                "token"
            ]
        ],
        on=[
            "country",
            "token"
        ],
        how="inner",
        suffixes=(
            "_s1",
            "_target"
        )
    )

    candidates = candidates.rename(
        columns={
            "entity_id_s1": "s1_id",
            "entity_id_target": "target_id"
        }
    )

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    candidates = candidates[
        [
            "s1_id",
            "target_id"
        ]
    ].drop_duplicates()

    print()
    print(
        "Unique candidate pairs:",
        len(candidates)
    )

    # --------------------------------------------------
    # Candidate counts per S1
    # --------------------------------------------------

    counts = (
        candidates
        .groupby("s1_id")
        .size()
    )

    # Include S1 entities with zero candidates
    counts = counts.reindex(
        s1["entity_id"],
        fill_value=0
    )

    print()
    print(
        "Candidate statistics:"
    )

    print(
        counts.describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                0.999
            ]
        )
    )

    print()
    print(
        "S1 with 0 candidates:",
        (counts == 0).sum()
    )

    print(
        "S1 with >100 candidates:",
        (counts > 100).sum()
    )

    print(
        "S1 with >500 candidates:",
        (counts > 500).sum()
    )

    print(
        "S1 with >1000 candidates:",
        (counts > 1000).sum()
    )

    print(
        "S1 with >5000 candidates:",
        (counts > 5000).sum()
    )

    print()
    print(
        "Average candidates per S1:",
        f"{counts.mean():.2f}"
    )

    return len(candidates)


def main():

    print("Loading TEST data...")

    s1, s2, s3 = load_test_data()

    s2_count = analyze_target(
        s1,
        s2,
        "S2"
    )

    s3_count = analyze_target(
        s1,
        s3,
        "S3"
    )

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        "S2 candidate pairs:",
        s2_count
    )

    print(
        "S3 candidate pairs:",
        s3_count
    )

    print(
        "Total candidate pairs:",
        s2_count + s3_count
    )


if __name__ == "__main__":
    main()