from data_loader import load_training_data


def parse_ids(value):

    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def inspect_s1(
    s1_id,
    source1,
    source2,
    source3,
    ground_truth
):

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    s1 = source1[
        source1["entity_id"] == s1_id
    ]

    print("\n==============================")
    print("SOURCE 1")
    print("==============================")

    print(
        s1.to_string(index=False)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    if gt.empty:
        print("No ground truth found.")
        return

    matched_ids = parse_ids(
        gt.iloc[0]["matched_entity_ids"]
    )

    print("\n==============================")
    print("TRUE MATCH IDS")
    print("==============================")

    print(matched_ids)

    # --------------------------------------------------------
    # S2 matches
    # --------------------------------------------------------

    s2_ids = [
        x for x in matched_ids
        if x.startswith("S2-")
    ]

    print("\n==============================")
    print("SOURCE 2 TRUE MATCHES")
    print("==============================")

    print(
        source2[
            source2["entity_id"].isin(s2_ids)
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # S3 matches
    # --------------------------------------------------------

    s3_ids = [
        x for x in matched_ids
        if x.startswith("S3-")
    ]

    print("\n==============================")
    print("SOURCE 3 TRUE MATCHES")
    print("==============================")

    print(
        source3[
            source3["entity_id"].isin(s3_ids)
        ].to_string(index=False)
    )


def main():

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    # Pick an entity with several matches
    example = ground_truth.iloc[100]

    inspect_s1(
        example["source1_entity_id"],
        source1,
        source2,
        source3,
        ground_truth
    )


if __name__ == "__main__":
    main()