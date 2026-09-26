from data_loader import load_training_data


def count_matches(value):

    if value is None:
        return 0

    value = str(value).strip()

    if not value:
        return 0

    return len([
        x for x in value.split(",")
        if x.strip()
    ])


def main():

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    # ========================================================
    # BASIC DATASET INFORMATION
    # ========================================================

    print("\n==============================")
    print("DATASET SHAPES")
    print("==============================")

    print("Source 1:", source1.shape)
    print("Source 2:", source2.shape)
    print("Source 3:", source3.shape)
    print("Ground Truth:", ground_truth.shape)

    # ========================================================
    # COLUMNS
    # ========================================================

    print("\n==============================")
    print("COLUMNS")
    print("==============================")

    print("Source 1:", source1.columns.tolist())
    print("Source 2:", source2.columns.tolist())
    print("Source 3:", source3.columns.tolist())
    print("Ground Truth:", ground_truth.columns.tolist())

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")

    for name, df in [
        ("Source 1", source1),
        ("Source 2", source2),
        ("Source 3", source3),
        ("Ground Truth", ground_truth),
    ]:

        print(f"\n{name}")
        print(df.isna().sum())

    # ========================================================
    # COUNTRIES
    # ========================================================

    print("\n==============================")
    print("COUNTRIES")
    print("==============================")

    for name, df in [
        ("Source 1", source1),
        ("Source 2", source2),
        ("Source 3", source3),
    ]:

        print(f"\n{name}")
        print(
            df["country"]
            .value_counts(dropna=False)
        )

    # ========================================================
    # MATCH COUNT
    # ========================================================

    ground_truth["match_count"] = (
        ground_truth["matched_entity_ids"]
        .apply(count_matches)
    )

    print("\n==============================")
    print("MATCH COUNT DISTRIBUTION")
    print("==============================")

    print(
        ground_truth["match_count"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SINGLETONS
    # ========================================================

    singleton_count = (
        ground_truth["match_count"] == 0
    ).sum()

    print("\n==============================")
    print("SINGLETONS")
    print("==============================")

    print(
        "Singleton count:",
        singleton_count
    )

    print(
        "Singleton percentage:",
        round(
            singleton_count /
            len(ground_truth) * 100,
            2
        ),
        "%"
    )

    # ========================================================
    # MULTIPLE MATCHES
    # ========================================================

    multiple_count = (
        ground_truth["match_count"] > 1
    ).sum()

    print("\n==============================")
    print("MULTIPLE MATCHES")
    print("==============================")

    print(
        "Entities with multiple matches:",
        multiple_count
    )

    print(
        "Maximum matches for one S1:",
        ground_truth["match_count"].max()
    )

    # ========================================================
    # DUPLICATE IDS
    # ========================================================

    print("\n==============================")
    print("DUPLICATE IDS")
    print("==============================")

    print(
        "Source 1:",
        source1["entity_id"].duplicated().sum()
    )

    print(
        "Source 2:",
        source2["entity_id"].duplicated().sum()
    )

    print(
        "Source 3:",
        source3["entity_id"].duplicated().sum()
    )

    print(
        "Ground Truth S1 IDs:",
        ground_truth[
            "source1_entity_id"
        ].duplicated().sum()
    )


if __name__ == "__main__":
    main()