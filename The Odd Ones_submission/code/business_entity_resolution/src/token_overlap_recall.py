import re
import pandas as pd
from data_loader import load_training_data


STOPWORDS = {
    "the", "and", "for", "inc", "incorporated",
    "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "pvt", "private", "public",
    "plc", "llp",
    "group", "services", "service", "solutions",
    "industries", "industry", "enterprise", "enterprises",
    "international", "global",
    "health", "healthcare", "medical", "medicine",
    "hospital", "clinic", "care", "center", "centre",
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "building", "block", "floor",
    "sector", "district", "city", "state",
    "india", "united", "states", "usa"
}


def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def tokenize(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
        and token not in STOPWORDS
    }


def parse_ground_truth(gt):

    gt = gt[
        gt["matched_entity_ids"].notna()
    ][
        ["source1_entity_id", "matched_entity_ids"]
    ].copy()

    pairs = (
        gt.assign(
            target_id=gt["matched_entity_ids"].str.split(",")
        )
        .explode("target_id")
    )

    pairs["target_id"] = (
        pairs["target_id"]
        .astype(str)
        .str.strip()
    )

    pairs = pairs.rename(
        columns={
            "source1_entity_id": "s1_id"
        }
    )

    return pairs[
        ["s1_id", "target_id"]
    ]


def main():

    print("Loading data...")

    s1, s2, s3, gt = load_training_data()

    print("Normalizing...")

    for df in [s1, s2, s3]:

        df["name_norm"] = (
            df["business_name"]
            .map(normalize_text)
        )

        df["address_norm"] = (
            df["business_address"]
            .map(normalize_text)
        )

        df["name_tokens"] = (
            df["name_norm"]
            .map(tokenize)
        )

        df["address_tokens"] = (
            df["address_norm"]
            .map(tokenize)
        )

    print("Building target lookup...")

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    target_lookup = target.set_index(
        "entity_id"
    )

    s1_lookup = s1.set_index(
        "entity_id"
    )

    print("Building true pairs...")

    pairs = parse_ground_truth(gt)

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    pairs = pairs.sample(
        n=min(200_000, len(pairs)),
        random_state=42
    )

    print(
        f"Pairs analyzed: {len(pairs):,}"
    )

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    thresholds = [
        1,
        2,
        3,
        4,
        5
    ]

    name_hits = {
        x: 0 for x in thresholds
    }

    address_hits = {
        x: 0 for x in thresholds
    }

    combined_hits = {
        x: 0 for x in thresholds
    }

    total = 0

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    for row in pairs.itertuples(
        index=False
    ):

        s1_row = s1_lookup.loc[
            row.s1_id
        ]

        target_row = target_lookup.loc[
            row.target_id
        ]

        name_shared = (
            s1_row["name_tokens"]
            &
            target_row["name_tokens"]
        )

        address_shared = (
            s1_row["address_tokens"]
            &
            target_row["address_tokens"]
        )

        name_count = len(name_shared)
        address_count = len(address_shared)

        total += 1

        for threshold in thresholds:

            if name_count >= threshold:
                name_hits[threshold] += 1

            if address_count >= threshold:
                address_hits[threshold] += 1

            if (
                name_count >= threshold
                or
                address_count >= threshold
            ):
                combined_hits[threshold] += 1

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("TOKEN OVERLAP RECALL")
    print("=" * 65)

    for threshold in thresholds:

        print()
        print(
            f"Shared tokens >= {threshold}"
        )

        print(
            f"Name:     "
            f"{name_hits[threshold] / total * 100:.2f}%"
        )

        print(
            f"Address:  "
            f"{address_hits[threshold] / total * 100:.2f}%"
        )

        print(
            f"Combined: "
            f"{combined_hits[threshold] / total * 100:.2f}%"
        )


if __name__ == "__main__":
    main()