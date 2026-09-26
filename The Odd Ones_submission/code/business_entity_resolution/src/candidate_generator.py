import re
from collections import defaultdict
import pandas as pd

from data_loader import load_training_data


# ============================================================
# CONFIG
# ============================================================

STOPWORDS = {
    # Business/legal
    "the", "and", "for", "inc", "incorporated",
    "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "pvt", "private", "public",
    "plc", "llp",

    # Generic business
    "group", "services", "service", "solutions",
    "industries", "industry", "enterprise", "enterprises",
    "international", "global",

    # Healthcare / generic
    "health", "healthcare", "medical", "medicine",
    "hospital", "clinic", "care", "center", "centre",

    # Address
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "building", "block", "floor",
    "sector", "district", "city", "state",

    # Common geographic words
    "india", "united", "states", "usa"
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokenize(text):
    if not text:
        return set()

    tokens = text.split()

    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS
    }


def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)

    df["name_tokens"] = df["name_norm"].map(tokenize)
    df["address_tokens"] = df["address_norm"].map(tokenize)

    return df


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(target_df):

    print("Building target indexes...")

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in target_df.itertuples(index=False):

        entity_id = row.entity_id
        country = row.country

        # Name tokens
        for token in row.name_tokens:

            key = (country, token)

            name_index[key].append(entity_id)

        # Address tokens
        for token in row.address_tokens:

            key = (country, token)

            address_index[key].append(entity_id)

    print(f"Name index keys: {len(name_index):,}")
    print(f"Address index keys: {len(address_index):,}")

    return name_index, address_index


# ============================================================
# CANDIDATES FOR ONE S1
# ============================================================

def generate_candidates_for_row(
    row,
    name_index,
    address_index
):

    country = row.country

    candidates = set()

    # --------------------------------------------------------
    # NAME BLOCK
    # --------------------------------------------------------

    for token in row.name_tokens:

        key = (country, token)

        candidates.update(name_index.get(key, []))

    # --------------------------------------------------------
    # ADDRESS BLOCK
    # --------------------------------------------------------

    for token in row.address_tokens:

        key = (country, token)

        candidates.update(address_index.get(key, []))

    return candidates


# ============================================================
# TEST CANDIDATE RECALL
# ============================================================
def parse_ground_truth(gt):

    # Only keep rows that actually have matches.
    # Do NOT create a Python list for every S1 row.
    gt = gt[
        gt["matched_entity_ids"].notna()
        & (gt["matched_entity_ids"].astype(str).str.strip() != "")
    ][
        ["source1_entity_id", "matched_entity_ids"]
    ].copy()

    # Split and explode directly.
    pairs = (
        gt.assign(
            target_id=gt["matched_entity_ids"].str.split(",")
        )
        .explode("target_id", ignore_index=False)
    )

    pairs["target_id"] = pairs["target_id"].str.strip()

    pairs = pairs.rename(
        columns={
            "source1_entity_id": "s1_id"
        }
    )

    return pairs[
        ["s1_id", "target_id"]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_training_data()

    # ========================================================
    # PREPARE DATA
    # ========================================================

    print("Preparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print("Building target dataframe...")

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    print(f"S1 records: {len(s1):,}")
    print(f"Target records: {len(target):,}")

    # ========================================================
    # BUILD TARGET INDEX
    # ========================================================

    name_index, address_index = build_index(target)

    # ========================================================
    # SAMPLE S1
    # ========================================================

    sample_ids = (
        s1["entity_id"]
        .sample(
            n=min(10000, len(s1)),
            random_state=42
        )
        .tolist()
    )

    sample_id_set = set(sample_ids)

    print(
        f"Selected {len(sample_ids):,} S1 entities"
    )

    # ========================================================
    # BUILD TRUE MATCH LOOKUP ONLY FOR SAMPLE
    # ========================================================

    print("Reading ground truth for sample...")

    gt_sample = gt[
        gt["source1_entity_id"].isin(sample_id_set)
    ][
        ["source1_entity_id", "matched_entity_ids"]
    ].copy()

    true_matches = defaultdict(set)

    for row in gt_sample.itertuples(index=False):

        if pd.isna(row.matched_entity_ids):
            continue

        value = str(row.matched_entity_ids).strip()

        if not value:
            continue

        for target_id in value.split(","):

            target_id = target_id.strip()

            if target_id:
                true_matches[
                    row.source1_entity_id
                ].add(target_id)

    print(
        f"S1 entities with matches: "
        f"{len(true_matches):,}"
    )

    # ========================================================
    # LOOKUP
    # ========================================================

    s1_lookup = s1.set_index("entity_id")

    # ========================================================
    # TEST CANDIDATE GENERATION
    # ========================================================

    total_true = 0
    total_found = 0

    candidate_counts = []

    print()
    print("Testing candidate generation...")

    for i, s1_id in enumerate(sample_ids):

        row = s1_lookup.loc[s1_id]

        candidates = generate_candidates_for_row(
            row,
            name_index,
            address_index
        )

        truth = true_matches.get(
            s1_id,
            set()
        )

        found = candidates.intersection(truth)

        total_true += len(truth)
        total_found += len(found)

        candidate_counts.append(
            len(candidates)
        )

        if (i + 1) % 1000 == 0:

            print(
                f"Processed "
                f"{i + 1:,}/"
                f"{len(sample_ids):,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    recall = (
        total_found / total_true
        if total_true
        else 0
    )

    candidate_series = pd.Series(
        candidate_counts
    )

    print()
    print("=" * 60)
    print("CANDIDATE GENERATOR TEST")
    print("=" * 60)

    print(
        f"True matches:       {total_true:,}"
    )

    print(
        f"Found by blocking:  {total_found:,}"
    )

    print(
        f"Candidate recall:   {recall * 100:.4f}%"
    )

    print()

    print(
        f"Mean candidates/S1: "
        f"{candidate_series.mean():.2f}"
    )

    print(
        f"Median candidates/S1: "
        f"{candidate_series.median():.2f}"
    )

    print(
        f"95th percentile: "
        f"{candidate_series.quantile(0.95):.2f}"
    )

    print(
        f"99th percentile: "
        f"{candidate_series.quantile(0.99):.2f}"
    )

    print(
        f"Maximum candidates: "
        f"{candidate_series.max():,}"
    )


if __name__ == "__main__":
    main()