import re
import pandas as pd
from collections import defaultdict

from data_loader import load_training_data


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "and", "for", "inc", "incorporated",
    "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "pvt", "private", "public",
    "plc", "llp",

    "group", "services", "service", "solutions",
    "industries", "industry", "enterprise",
    "enterprises", "international", "global",

    "health", "healthcare", "medical", "medicine",
    "hospital", "clinic", "care", "center", "centre",

    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "building", "block", "floor",
    "sector", "district", "city", "state",

    "india", "united", "states", "usa"
}


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ground_truth(gt):

    gt = gt[
        gt["matched_entity_ids"].notna()
        & (
            gt["matched_entity_ids"]
            .astype(str)
            .str.strip()
            != ""
        )
    ][
        [
            "source1_entity_id",
            "matched_entity_ids"
        ]
    ].copy()

    pairs = (
        gt.assign(
            target_id=
            gt["matched_entity_ids"].str.split(",")
        )
        .explode("target_id")
    )

    pairs["target_id"] = (
        pairs["target_id"]
        .astype(str)
        .str.strip()
    )

    return pairs[
        [
            "source1_entity_id",
            "target_id"
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

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

    # ========================================================
    # TARGET
    # ========================================================

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

    # ========================================================
    # TOKEN FREQUENCIES
    # ========================================================

    print("Building token frequencies...")

    token_frequency = defaultdict(int)

    for row in target.itertuples(
        index=False
    ):

        tokens = (
            row.name_tokens
            |
            row.address_tokens
        )

        for token in tokens:

            token_frequency[
                (row.country, token)
            ] += 1

    print(
        f"Unique token keys: "
        f"{len(token_frequency):,}"
    )

    # ========================================================
    # TRUE PAIRS
    # ========================================================

    print("Building true pairs...")

    pairs = parse_ground_truth(gt)

    pairs = pairs.sample(
        n=min(200_000, len(pairs)),
        random_state=42
    )

    print(
        f"Pairs analyzed: "
        f"{len(pairs):,}"
    )

    # ========================================================
    # TEST DIFFERENT CUT-OFFS
    # ========================================================

    CUTOFFS = [
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000
    ]

    hits = {
        cutoff: 0
        for cutoff in CUTOFFS
    }

    total = len(pairs)

    # ========================================================
    # ANALYZE
    # ========================================================

    for i, row in enumerate(
        pairs.itertuples(index=False)
    ):

        s1_row = s1_lookup.loc[
            row.source1_entity_id
        ]

        target_row = target_lookup.loc[
            row.target_id
        ]

        s1_tokens = (
            s1_row["name_tokens"]
            |
            s1_row["address_tokens"]
        )

        target_tokens = (
            target_row["name_tokens"]
            |
            target_row["address_tokens"]
        )

        shared = (
            s1_tokens
            &
            target_tokens
        )

        if not shared:
            continue

        # ----------------------------------------------------
        # Frequencies of shared tokens
        # ----------------------------------------------------

        frequencies = [
            token_frequency.get(
                (
                    s1_row["country"],
                    token
                ),
                999999999
            )
            for token in shared
        ]

        # Rarest shared token
        rarest_frequency = min(
            frequencies
        )

        # Number of shared tokens
        shared_count = len(shared)

        # ----------------------------------------------------
        # Hybrid rule:
        #
        # rare shared token
        # OR
        # >= 2 shared tokens
        # ----------------------------------------------------

        for cutoff in CUTOFFS:

            if (
                rarest_frequency <= cutoff
                or
                shared_count >= 2
            ):

                hits[cutoff] += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("HYBRID BLOCK RECALL")
    print("=" * 70)

    for cutoff in CUTOFFS:

        recall = (
            hits[cutoff]
            / total
            * 100
        )

        print(
            f"Rare token <= {cutoff:4d}"
            f" OR shared tokens >= 2:"
            f" {recall:.2f}%"
        )


if __name__ == "__main__":
    main()