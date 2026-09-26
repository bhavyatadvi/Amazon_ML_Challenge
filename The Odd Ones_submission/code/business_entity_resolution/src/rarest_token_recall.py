import re
import pandas as pd

from data_loader import load_training_data


STOPWORDS = {
    "the", "and", "of", "for", "in", "on", "at", "to",
    "a", "an",

    "limited", "ltd", "llp", "inc", "incorporated",
    "corp", "corporation", "company", "co",
    "private", "pvt", "plc",

    "group", "center", "centre", "clinic",
    "services", "service", "health", "medical",
}


def normalize_text_series(series):

    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
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


def build_frequency(df, column):

    frequency = {}

    for row in df[
        ["country", column]
    ].itertuples(index=False):

        tokens = tokenize(row[1])

        for token in tokens:

            key = (
                row[0],
                token
            )

            frequency[key] = (
                frequency.get(key, 0) + 1
            )

    return frequency


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Normalizing...")

    for df in [s1, s2, s3]:

        df["name_norm"] = normalize_text_series(
            df["business_name"]
        )

        df["address_norm"] = normalize_text_series(
            df["business_address"]
        )

    print("Building true pairs...")

    pairs = parse_ground_truth(gt)

    # Sample for fast experimentation
    pairs = pairs.sample(
        n=min(200_000, len(pairs)),
        random_state=42
    )

    print(
        "Pairs analyzed:",
        len(pairs)
    )

    # --------------------------------------------------
    # Target data
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Frequencies
    # --------------------------------------------------

    print("Building name frequencies...")

    name_frequency = build_frequency(
        target,
        "name_norm"
    )

    print(
        "Unique name tokens:",
        len(name_frequency)
    )

    print("Building address frequencies...")

    address_frequency = build_frequency(
        target,
        "address_norm"
    )

    print(
        "Unique address tokens:",
        len(address_frequency)
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    total = len(pairs)

    CUTOFFS = [50, 100, 250, 500, 1000, 2500, 5000]

    name_hits = {cutoff: 0 for cutoff in CUTOFFS}
    address_hits = {cutoff: 0 for cutoff in CUTOFFS}
    combined_hits = {cutoff: 0 for cutoff in CUTOFFS}

    for row in pairs.itertuples(
        index=False
    ):

        s1_row = s1_lookup.loc[row.s1_id]
        target_row = target_lookup.loc[row.target_id]
        country = s1_row["country"]

        # ------------------------------
        # NAME
        # ------------------------------

        s1_tokens = tokenize(s1_row["name_norm"])
        target_tokens = tokenize(target_row["name_norm"])
        shared_name = s1_tokens & target_tokens

        min_name_frequency = 999999999
        if shared_name:
            min_name_frequency = min(
                name_frequency.get((country, token), 999999999)
                for token in shared_name
            )

        # ------------------------------
        # ADDRESS
        # ------------------------------

        s1_tokens = tokenize(s1_row["address_norm"])
        target_tokens = tokenize(target_row["address_norm"])
        shared_address = s1_tokens & target_tokens

        min_address_frequency = 999999999
        if shared_address:
            min_address_frequency = min(
                address_frequency.get((country, token), 999999999)
                for token in shared_address
            )

        # ------------------------------
        # UPDATE HITS
        # ------------------------------

        for cutoff in CUTOFFS:
            name_found = (min_name_frequency <= cutoff)
            address_found = (min_address_frequency <= cutoff)

            if name_found:
                name_hits[cutoff] += 1
            if address_found:
                address_hits[cutoff] += 1
            if name_found or address_found:
                combined_hits[cutoff] += 1

    print()
    print("=" * 60)
    print("RAREST TOKEN RECALL")
    print("=" * 60)

    for cutoff in CUTOFFS:
        print()
        print(f"Cutoff: <= {cutoff}")
        print("Name:", f"{name_hits[cutoff] / total:.2%}")
        print("Address:", f"{address_hits[cutoff] / total:.2%}")
        print("Combined:", f"{combined_hits[cutoff] / total:.2%}")


if __name__ == "__main__":
    main()