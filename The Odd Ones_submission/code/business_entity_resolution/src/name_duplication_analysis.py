import pandas as pd
import numpy as np
import re

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


def analyze_source(df, source_name):

    print()
    print("=" * 60)
    print(source_name)
    print("=" * 60)

    temp = df[
        ["entity_id", "business_name", "country"]
    ].copy()

    temp["name_norm"] = normalize_text_series(
        temp["business_name"]
    )

    # Empty names
    empty = (temp["name_norm"] == "").sum()

    print()
    print("Empty normalized names:", empty)

    # Remove empty names for duplication analysis
    non_empty = temp[
        temp["name_norm"] != ""
    ]

    # Frequency of each normalized name
    counts = (
        non_empty
        .groupby(
            ["country", "name_norm"],
            sort=False
        )
        .size()
    )

    print()
    print("Unique normalized names:", len(counts))

    print()
    print("Total records with non-empty names:", len(non_empty))

    # --------------------------------------------------
    # Frequency distribution
    # --------------------------------------------------

    print()
    print("NAME FREQUENCY DISTRIBUTION")
    print()

    frequency_distribution = (
        counts
        .value_counts()
        .sort_index()
    )

    print(
        frequency_distribution.head(20)
    )

    # --------------------------------------------------
    # How many records have unique names?
    # --------------------------------------------------

    unique_name_records = (
        counts[counts == 1].sum()
    )

    print()
    print(
        "Records with unique normalized name:",
        unique_name_records
    )

    print(
        "Percentage:",
        f"{unique_name_records / len(non_empty):.2%}"
    )

    # --------------------------------------------------
    # Records belonging to duplicated names
    # --------------------------------------------------

    duplicated_records = (
        counts[counts > 1].sum()
    )

    print()
    print(
        "Records with duplicated normalized name:",
        duplicated_records
    )

    print(
        "Percentage:",
        f"{duplicated_records / len(non_empty):.2%}"
    )

    # --------------------------------------------------
    # Very common names
    # --------------------------------------------------

    print()
    print("TOP 20 MOST COMMON NORMALIZED NAMES")

    top = (
        counts
        .sort_values(ascending=False)
        .head(20)
    )

    print(top)

    # --------------------------------------------------
    # Candidate-size statistics
    # --------------------------------------------------

    print()
    print("CANDIDATE SIZE STATISTICS")

    print(
        counts.describe()
    )


def main():

    print("Loading training data...")

    s1, s2, s3, ground_truth = load_training_data()

    analyze_source(s1, "SOURCE 1")

    analyze_source(s2, "SOURCE 2")

    analyze_source(s3, "SOURCE 3")


if __name__ == "__main__":
    main()