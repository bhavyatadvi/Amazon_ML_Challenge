import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    average_precision_score
)
from sklearn.ensemble import HistGradientBoostingClassifier

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

TRAINING_PAIRS = os.path.join(
    BASE_DIR,
    "output",
    "other",
    "training_pairs.tsv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "output",
    "other",
    "matcher.joblib"
)


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


def tokens(value):

    if not value:
        return set()

    return set(
        x
        for x in value.split()
        if x
    )


# ============================================================
# STRING FEATURES
# ============================================================

def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


def overlap_ratio(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)

    return intersection / min(
        len(a),
        len(b)
    )


def char_similarity(a, b):

    if not a or not b:
        return 0.0

    # Cheap character-level similarity.
    # SequenceMatcher is avoided because millions
    # of comparisons can become expensive.
    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))

    if longer == 0:
        return 0.0

    # Character-set Jaccard
    ca = set(a.replace(" ", ""))
    cb = set(b.replace(" ", ""))

    if not ca or not cb:
        return 0.0

    return len(ca & cb) / len(ca | cb)


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    pairs,
    s1,
    target
):

    print("Preparing source data...")

    s1 = s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    target = target[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("Normalizing names and addresses...")

    for df in [s1, target]:

        df["name_norm"] = (
            df["business_name"]
            .map(normalize_text)
        )

        df["address_norm"] = (
            df["business_address"]
            .map(normalize_text)
        )

    # --------------------------------------------------------
    # Lookups
    # --------------------------------------------------------

    print("Building lookups...")

    s1_lookup = (
        s1
        .set_index("entity_id")
        [
            [
                "name_norm",
                "address_norm",
                "country"
            ]
        ]
        .to_dict("index")
    )

    target_lookup = (
        target
        .set_index("entity_id")
        [
            [
                "name_norm",
                "address_norm",
                "country"
            ]
        ]
        .to_dict("index")
    )

    # --------------------------------------------------------
    # Feature generation
    # --------------------------------------------------------

    records = []

    total = len(pairs)

    for i, row in enumerate(
        pairs.itertuples(index=False),
        start=1
    ):

        s1_id = row.source1_entity_id
        target_id = row.target_entity_id

        s = s1_lookup.get(s1_id)
        t = target_lookup.get(target_id)

        if s is None or t is None:
            continue

        name1 = s["name_norm"]
        name2 = t["name_norm"]

        addr1 = s["address_norm"]
        addr2 = t["address_norm"]

        name_tokens_1 = tokens(name1)
        name_tokens_2 = tokens(name2)

        addr_tokens_1 = tokens(addr1)
        addr_tokens_2 = tokens(addr2)

        name_jaccard = jaccard(
            name_tokens_1,
            name_tokens_2
        )

        address_jaccard = jaccard(
            addr_tokens_1,
            addr_tokens_2
        )

        name_overlap = overlap_ratio(
            name_tokens_1,
            name_tokens_2
        )

        address_overlap = overlap_ratio(
            addr_tokens_1,
            addr_tokens_2
        )

        record = {

            # -------------------------------
            # Exact features
            # -------------------------------

            "country_match":
                int(
                    s["country"]
                    ==
                    t["country"]
                ),

            "name_exact":
                int(
                    bool(name1)
                    and
                    name1 == name2
                ),

            "address_exact":
                int(
                    bool(addr1)
                    and
                    addr1 == addr2
                ),

            # -------------------------------
            # Token features
            # -------------------------------

            "name_jaccard":
                name_jaccard,

            "address_jaccard":
                address_jaccard,

            "name_overlap":
                name_overlap,

            "address_overlap":
                address_overlap,

            "name_shared_tokens":
                len(
                    name_tokens_1
                    &
                    name_tokens_2
                ),

            "address_shared_tokens":
                len(
                    addr_tokens_1
                    &
                    addr_tokens_2
                ),

            # -------------------------------
            # Character features
            # -------------------------------

            "name_char_similarity":
                char_similarity(
                    name1,
                    name2
                ),

            "address_char_similarity":
                char_similarity(
                    addr1,
                    addr2
                ),

            # -------------------------------
            # Length features
            # -------------------------------

            "name_length_diff":
                abs(
                    len(name1)
                    -
                    len(name2)
                ),

            "address_length_diff":
                abs(
                    len(addr1)
                    -
                    len(addr2)
                ),

            "name_length_ratio":
                (
                    min(len(name1), len(name2))
                    /
                    max(len(name1), len(name2))
                    if max(len(name1), len(name2))
                    else 0
                ),

            "address_length_ratio":
                (
                    min(len(addr1), len(addr2))
                    /
                    max(len(addr1), len(addr2))
                    if max(len(addr1), len(addr2))
                    else 0
                ),

            # -------------------------------
            # Target source
            # -------------------------------

            "target_is_s2":
                int(
                    str(target_id).startswith("S2-")
                ),

            "target_is_s3":
                int(
                    str(target_id).startswith("S3-")
                ),

            "label":
                int(row.label)
        }

        records.append(record)

        if i % 10000 == 0:

            print(
                f"Features: "
                f"{i:,}/{total:,}"
            )

    result = pd.DataFrame(records)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAIN MATCHER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training pairs
    # --------------------------------------------------------

    print()
    print("Loading training pairs...")

    pairs = pd.read_csv(
        TRAINING_PAIRS,
        sep="\t"
    )

    print(
        f"Training pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Load original data
    # --------------------------------------------------------

    print()
    print("Loading source data...")

    s1, s2, s3, gt = (
        load_training_data()
    )

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    print(
        f"S1: {len(s1):,}"
    )

    print(
        f"Target: {len(target):,}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    print()
    print("Building features...")

    features = build_features(
        pairs,
        s1,
        target
    )

    print()
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in features.columns
        if c != "label"
    ]

    X = features[
        feature_columns
    ]

    y = features[
        "label"
    ]

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print()
    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training model...")

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("Evaluating...")

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # Default threshold for initial inspection.
    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    average_precision = (
        average_precision_score(
            y_valid,
            probabilities
        )
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Precision @ 0.50: "
        f"{precision:.6f}"
    )

    print(
        f"Recall @ 0.50:    "
        f"{recall:.6f}"
    )

    print(
        f"F1 @ 0.50:        "
        f"{f1:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    print()
    print(
        classification_report(
            y_valid,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    artifact = {
        "model": model,
        "features": feature_columns,
        "threshold": threshold
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()