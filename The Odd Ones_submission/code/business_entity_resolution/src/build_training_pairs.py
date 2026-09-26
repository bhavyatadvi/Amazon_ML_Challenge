import os
import re
import random
from collections import defaultdict

import pandas as pd

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "output",
    "other"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "training_pairs.tsv"
)

# Number of S1 entities used for training
TRAIN_S1 = 20000

# Negative candidates per positive pair
NEGATIVES_PER_POSITIVE = 3


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = re.sub(
        r"[^a-z0-9]+",
        " ",
        x
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    )

    return x.strip()


# ============================================================
# PARSE GROUND TRUTH
# ============================================================

def parse_ground_truth(gt):

    truth = defaultdict(set)

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if pd.isna(row.matched_entity_ids):
            continue

        value = str(
            row.matched_entity_ids
        ).strip()

        if not value:
            continue

        # Ground truth uses comma-separated IDs
        for target_id in value.split(","):

            target_id = target_id.strip()

            if target_id:
                truth[s1_id].add(target_id)

    return truth


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    print("Preparing target data...")

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Target lookup
    # --------------------------------------------------------

    target_lookup = (
        target[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ]
        .copy()
    )

    target_lookup["name_norm"] = (
        target_lookup["business_name"]
        .map(normalize_text)
    )

    target_lookup["address_norm"] = (
        target_lookup["business_address"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # Build exact indexes
    # --------------------------------------------------------

    print("Building exact indexes...")

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in target_lookup.itertuples(index=False):

        if row.name_norm:

            name_index[
                (row.country, row.name_norm)
            ].append(row.entity_id)

        if row.address_norm:

            address_index[
                (row.country, row.address_norm)
            ].append(row.entity_id)

    print(
        "Name index:",
        len(name_index)
    )

    print(
        "Address index:",
        len(address_index)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("Parsing ground truth...")

    truth = parse_ground_truth(gt)

    # --------------------------------------------------------
    # Select S1 training sample
    # --------------------------------------------------------

    available = [
        x
        for x in truth
        if truth[x]
    ]

    random.seed(42)

    random.shuffle(
        available
    )

    selected = available[
        :TRAIN_S1
    ]

    selected_set = set(
        selected
    )

    print(
        "Selected S1:",
        len(selected)
    )

    # --------------------------------------------------------
    # S1 lookup
    # --------------------------------------------------------

    s1_lookup = (
        s1[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ]
        .copy()
    )

    s1_lookup["name_norm"] = (
        s1_lookup["business_name"]
        .map(normalize_text)
    )

    s1_lookup["address_norm"] = (
        s1_lookup["business_address"]
        .map(normalize_text)
    )

    s1_lookup = (
        s1_lookup
        .set_index("entity_id")
    )

    # --------------------------------------------------------
    # Build pairs
    # --------------------------------------------------------

    records = []

    processed = 0

    for s1_id in selected:

        row = s1_lookup.loc[s1_id]

        true_ids = truth[s1_id]

        # ----------------------------------------------------
        # Candidate discovery
        # ----------------------------------------------------

        candidates = set()

        name_key = (
            row["country"],
            row["name_norm"]
        )

        address_key = (
            row["country"],
            row["address_norm"]
        )

        if row["name_norm"]:

            candidates.update(
                name_index.get(
                    name_key,
                    []
                )
            )

        if row["address_norm"]:

            candidates.update(
                address_index.get(
                    address_key,
                    []
                )
            )

        # ----------------------------------------------------
        # Positive pairs
        # ----------------------------------------------------

        for target_id in true_ids:

            records.append({

                "source1_entity_id":
                    s1_id,

                "target_entity_id":
                    target_id,

                "label":
                    1
            })

        # ----------------------------------------------------
        # Negative pairs
        # ----------------------------------------------------

        negative_candidates = [
            x
            for x in candidates
            if x not in true_ids
        ]

        random.shuffle(
            negative_candidates
        )

        max_negative = (
            len(true_ids)
            *
            NEGATIVES_PER_POSITIVE
        )

        negative_candidates = (
            negative_candidates[
                :max_negative
            ]
        )

        for target_id in negative_candidates:

            records.append({

                "source1_entity_id":
                    s1_id,

                "target_entity_id":
                    target_id,

                "label":
                    0
            })

        processed += 1

        if processed % 1000 == 0:

            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(selected):,}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = pd.DataFrame(
        records
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    print()
    print("=" * 60)
    print("TRAINING PAIRS")
    print("=" * 60)

    print(
        "Rows:",
        len(result)
    )

    print(
        "Positive:",
        int(
            (result["label"] == 1).sum()
        )
    )

    print(
        "Negative:",
        int(
            (result["label"] == 0).sum()
        )
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()