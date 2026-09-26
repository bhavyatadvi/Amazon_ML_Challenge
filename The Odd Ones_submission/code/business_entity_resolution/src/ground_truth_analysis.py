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


def main():

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    # --------------------------------------------------------
    # Parse matched IDs
    # --------------------------------------------------------

    ground_truth["matched_list"] = (
        ground_truth["matched_entity_ids"]
        .apply(parse_ids)
    )

    # --------------------------------------------------------
    # Count S2 and S3 matches
    # --------------------------------------------------------

    def count_source(ids, prefix):
        return sum(
            entity_id.startswith(prefix)
            for entity_id in ids
        )

    ground_truth["s2_count"] = (
        ground_truth["matched_list"]
        .apply(lambda x: count_source(x, "S2-"))
    )

    ground_truth["s3_count"] = (
        ground_truth["matched_list"]
        .apply(lambda x: count_source(x, "S3-"))
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\n==============================")
    print("S2 MATCH COUNT DISTRIBUTION")
    print("==============================")

    print(
        ground_truth["s2_count"]
        .value_counts()
        .sort_index()
    )

    print("\n==============================")
    print("S3 MATCH COUNT DISTRIBUTION")
    print("==============================")

    print(
        ground_truth["s3_count"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Match source combinations
    # --------------------------------------------------------

    def match_type(row):

        s2 = row["s2_count"]
        s3 = row["s3_count"]

        if s2 > 0 and s3 > 0:
            return "BOTH_S2_S3"

        if s2 > 0:
            return "S2_ONLY"

        if s3 > 0:
            return "S3_ONLY"

        return "NONE"

    ground_truth["match_type"] = (
        ground_truth.apply(
            match_type,
            axis=1
        )
    )

    print("\n==============================")
    print("MATCH TYPE")
    print("==============================")

    print(
        ground_truth["match_type"]
        .value_counts()
    )

    def country_match_analysis(
        source1,
        ground_truth
    ):

        merged = ground_truth.merge(
            source1[
                ["entity_id", "country"]
            ],
            left_on="source1_entity_id",
            right_on="entity_id",
            how="left"
        )

        print("\n==============================")
        print("MATCH COUNT BY COUNTRY")
        print("==============================")

        print(
            merged.groupby("country")["match_count"]
            .agg([
                "count",
                "mean",
                "median",
                "min",
                "max"
            ])
        )

    ground_truth["match_count"] = (
        ground_truth["matched_entity_ids"]
        .apply(
            lambda x: len(parse_ids(x))
        )
    )

    country_match_analysis(
        source1,
        ground_truth
    )


if __name__ == "__main__":
    main()