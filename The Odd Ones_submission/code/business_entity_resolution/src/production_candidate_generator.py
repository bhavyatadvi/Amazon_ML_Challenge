import os
import re
from collections import defaultdict, Counter

import pandas as pd

from data_loader import load_test_data


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "candidate_pairs.tsv"
)

# Ignore tokens occurring more than this many times.
# Extremely common tokens create huge candidate sets.
MAX_TOKEN_FREQUENCY = 5000

# Keep only the rarest N useful tokens from each field.
RAREST_NAME_TOKENS = 3
RAREST_ADDRESS_TOKENS = 4

# Maximum candidates retained per S1.
MAX_CANDIDATES_PER_S1 = 5000

# Progress reporting.
PRINT_EVERY = 10000


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


def get_tokens(value):

    if not value:
        return set()

    return set(
        token
        for token in value.split()
        if token
    )


# ============================================================
# BUILD TOKEN INDEX
# ============================================================

def build_token_index(target, column):

    print(
        f"Building {column} token index..."
    )

    index = defaultdict(list)

    frequencies = Counter()

    # --------------------------------------------------------
    # First pass: frequency
    # --------------------------------------------------------

    for value in target[column]:

        tokens = get_tokens(value)

        for token in tokens:

            frequencies[token] += 1

    print(
        f"Unique {column} tokens:",
        len(frequencies)
    )

    # --------------------------------------------------------
    # Second pass: index
    # --------------------------------------------------------

    for row in target.itertuples(index=False):

        value = getattr(
            row,
            column
        )

        tokens = get_tokens(value)

        for token in tokens:

            if (
                frequencies[token]
                <= MAX_TOKEN_FREQUENCY
            ):

                index[token].append(
                    row.entity_id
                )

    print(
        f"{column} index keys:",
        len(index)
    )

    return index, frequencies


# ============================================================
# CANDIDATE RANKING
# ============================================================

def rank_candidates(
    s1_row,
    candidates,
    target_lookup,
    name_freq,
    address_freq
):

    s1_name_tokens = get_tokens(
        s1_row.name_norm
    )

    s1_address_tokens = get_tokens(
        s1_row.address_norm
    )

    scored = []

    for target_id in candidates:

        target_row = target_lookup.get(
            target_id
        )

        if target_row is None:
            continue

        # Country should always agree.
        if (
            s1_row.country
            != target_row["country"]
        ):
            continue

        target_name_tokens = get_tokens(
            target_row["name_norm"]
        )

        target_address_tokens = get_tokens(
            target_row["address_norm"]
        )

        shared_name = (
            s1_name_tokens
            &
            target_name_tokens
        )

        shared_address = (
            s1_address_tokens
            &
            target_address_tokens
        )

        # ----------------------------------------------------
        # Exact matches
        # ----------------------------------------------------

        exact_name = int(
            bool(s1_row.name_norm)
            and
            s1_row.name_norm
            ==
            target_row["name_norm"]
        )

        exact_address = int(
            bool(s1_row.address_norm)
            and
            s1_row.address_norm
            ==
            target_row["address_norm"]
        )

        # ----------------------------------------------------
        # Rarity score
        # ----------------------------------------------------

        name_rarity = 0.0

        for token in shared_name:

            freq = name_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if freq > 0:

                name_rarity += (
                    1.0 / freq
                )

        address_rarity = 0.0

        for token in shared_address:

            freq = address_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if freq > 0:

                address_rarity += (
                    1.0 / freq
                )

        # ----------------------------------------------------
        # Retrieval score
        # ----------------------------------------------------

        score = (

            exact_name * 100000

            +

            exact_address * 100000

            +

            len(shared_name) * 100

            +

            len(shared_address) * 150

            +

            name_rarity * 10000

            +

            address_rarity * 10000
        )

        scored.append(
            (
                score,
                target_id
            )
        )

    scored.sort(
        reverse=True
    )

    return [
        target_id
        for _, target_id
        in scored[
            :MAX_CANDIDATES_PER_S1
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRODUCTION CANDIDATE GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print()
    print("Loading TEST data...")

    s1, s2, s3 = load_test_data()

    print(
        "S1 records:",
        f"{len(s1):,}"
    )

    print(
        "S2 records:",
        f"{len(s2):,}"
    )

    print(
        "S3 records:",
        f"{len(s3):,}"
    )

    # --------------------------------------------------------
    # Combine targets
    # --------------------------------------------------------

    target = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print()
    print("Normalizing...")

    s1["name_norm"] = (
        s1["business_name"]
        .map(normalize_text)
    )

    s1["address_norm"] = (
        s1["business_address"]
        .map(normalize_text)
    )

    target["name_norm"] = (
        target["business_name"]
        .map(normalize_text)
    )

    target["address_norm"] = (
        target["business_address"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # Target lookup
    # --------------------------------------------------------

    print()
    print("Building target lookup...")

    target_lookup = {}

    for row in target.itertuples(index=False):

        target_lookup[row.entity_id] = {
            "name_norm":
                row.name_norm,

            "address_norm":
                row.address_norm,

            "country":
                row.country
        }

    # --------------------------------------------------------
    # Token indexes
    # --------------------------------------------------------

    print()

    name_index, name_freq = (
        build_token_index(
            target,
            "name_norm"
        )
    )

    print()

    address_index, address_freq = (
        build_token_index(
            target,
            "address_norm"
        )
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    print()
    print("Generating candidates...")

    output_rows = []

    total_candidates = 0
    max_candidates = 0

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        candidates = set()

        name_tokens = get_tokens(
            row.name_norm
        )

        address_tokens = get_tokens(
            row.address_norm
        )

        # ----------------------------------------------------
        # Name candidates
        # ----------------------------------------------------

        useful_name_tokens = []

        for token in name_tokens:

            freq = name_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if (
                freq
                <= MAX_TOKEN_FREQUENCY
            ):

                useful_name_tokens.append(
                    (
                        freq,
                        token
                    )
                )

        useful_name_tokens.sort()

        useful_name_tokens = (
            useful_name_tokens[
                :RAREST_NAME_TOKENS
            ]
        )

        for _, token in useful_name_tokens:

            candidates.update(
                name_index.get(
                    token,
                    []
                )
            )

        # ----------------------------------------------------
        # Address candidates
        # ----------------------------------------------------

        useful_address_tokens = []

        for token in address_tokens:

            freq = address_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if (
                freq
                <= MAX_TOKEN_FREQUENCY
            ):

                useful_address_tokens.append(
                    (
                        freq,
                        token
                    )
                )

        useful_address_tokens.sort()

        useful_address_tokens = (
            useful_address_tokens[
                :RAREST_ADDRESS_TOKENS
            ]
        )

        for _, token in useful_address_tokens:

            candidates.update(
                address_index.get(
                    token,
                    []
                )
            )

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        ranked = rank_candidates(
            row,
            candidates,
            target_lookup,
            name_freq,
            address_freq
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        for target_id in ranked:

            output_rows.append(
                (
                    row.entity_id,
                    target_id
                )
            )

        candidate_count = len(
            ranked
        )

        total_candidates += (
            candidate_count
        )

        max_candidates = max(
            max_candidates,
            candidate_count
        )

        if i % PRINT_EVERY == 0:

            mean = (
                total_candidates / i
            )

            print(
                f"Processed "
                f"{i:,}/"
                f"{len(s1):,} | "
                f"Mean candidates/S1: "
                f"{mean:,.1f} | "
                f"Current: "
                f"{candidate_count:,} | "
                f"Max: "
                f"{max_candidates:,}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Saving candidate_pairs.tsv...")

    result = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_candidates = (
        total_candidates
        /
        len(s1)
    )

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Candidate pairs:",
        f"{len(result):,}"
    )

    print(
        "Mean candidates/S1:",
        f"{mean_candidates:,.2f}"
    )

    print(
        "Maximum candidates/S1:",
        f"{max_candidates:,}"
    )

    print(
        "Output:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()