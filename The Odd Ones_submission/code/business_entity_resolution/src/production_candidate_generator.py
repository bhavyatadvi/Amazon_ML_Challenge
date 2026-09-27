import os
import re
from collections import defaultdict, Counter

import pandas as pd

from data_loader import load_test_data


# ============================================================
# CONFIGURATION
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

# ------------------------------------------------------------
# Blocking controls
# ------------------------------------------------------------

# Ignore extremely common tokens.
# Common tokens generate enormous candidate sets.
MAX_TOKEN_FREQUENCY = 1000

# Number of rarest useful tokens used from each field.
RAREST_NAME_TOKENS = 2
RAREST_ADDRESS_TOKENS = 2

# Hard safety limit.
MAX_CANDIDATES_PER_S1 = 500

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

    return {
        token
        for token in value.split()
        if token
    }


# ============================================================
# BUILD TOKEN INDEX
# ============================================================

def build_token_index(target, column):

    print(
        f"Building {column} token index..."
    )

    frequencies = Counter()

    # --------------------------------------------------------
    # Pass 1: token frequencies
    # --------------------------------------------------------

    for value in target[column]:

        for token in get_tokens(value):

            frequencies[token] += 1

    print(
        f"Unique {column} tokens:",
        f"{len(frequencies):,}"
    )

    # --------------------------------------------------------
    # Pass 2: token -> entity IDs
    # --------------------------------------------------------

    index = defaultdict(list)

    for row in target.itertuples(index=False):

        value = getattr(
            row,
            column
        )

        for token in get_tokens(value):

            if (
                frequencies[token]
                <= MAX_TOKEN_FREQUENCY
            ):

                index[token].append(
                    row.entity_id
                )

    print(
        f"{column} index keys:",
        f"{len(index):,}"
    )

    return index, frequencies


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    s1_row,
    target_row,
    name_freq,
    address_freq
):

    name1 = get_tokens(
        s1_row.name_norm
    )

    name2 = get_tokens(
        target_row["name_norm"]
    )

    address1 = get_tokens(
        s1_row.address_norm
    )

    address2 = get_tokens(
        target_row["address_norm"]
    )

    shared_name = name1 & name2
    shared_address = address1 & address2

    # --------------------------------------------------------
    # Exact matches
    # --------------------------------------------------------

    exact_name = (
        bool(s1_row.name_norm)
        and
        s1_row.name_norm
        ==
        target_row["name_norm"]
    )

    exact_address = (
        bool(s1_row.address_norm)
        and
        s1_row.address_norm
        ==
        target_row["address_norm"]
    )

    # --------------------------------------------------------
    # Rarity score
    # --------------------------------------------------------

    name_rarity = 0.0

    for token in shared_name:

        frequency = name_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1
        )

        if frequency > 0:

            name_rarity += (
                1.0 / frequency
            )

    address_rarity = 0.0

    for token in shared_address:

        frequency = address_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1
        )

        if frequency > 0:

            address_rarity += (
                1.0 / frequency
            )

    # --------------------------------------------------------
    # Candidate score
    # --------------------------------------------------------
    #
    # Exact matches dominate.
    # Address overlap gets slightly more weight because
    # addresses are generally more discriminative.
    #
    # Rarity prevents common tokens from dominating.
    # --------------------------------------------------------

    score = 0.0

    if exact_name:
        score += 100000

    if exact_address:
        score += 100000

    score += (
        len(shared_name)
        * 100
    )

    score += (
        len(shared_address)
        * 150
    )

    score += (
        name_rarity
        * 10000
    )

    score += (
        address_rarity
        * 10000
    )

    return score


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
    # Combine target sources
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

        target_lookup[
            row.entity_id
        ] = {
            "name_norm":
                row.name_norm,

            "address_norm":
                row.address_norm,

            "country":
                row.country
        }

    # --------------------------------------------------------
    # Build indexes
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
    # Candidate generation
    # --------------------------------------------------------

    print()
    print("Generating candidates...")
    print(
        f"Maximum candidates/S1: "
        f"{MAX_CANDIDATES_PER_S1}"
    )

    output_rows = []

    total_candidates = 0
    max_candidates = 0

    # --------------------------------------------------------
    # Process S1
    # --------------------------------------------------------

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        candidates = set()

        # ----------------------------------------------------
        # Name tokens
        # ----------------------------------------------------

        name_tokens = get_tokens(
            row.name_norm
        )

        useful_name_tokens = []

        for token in name_tokens:

            frequency = name_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if (
                frequency
                <= MAX_TOKEN_FREQUENCY
            ):

                useful_name_tokens.append(
                    (
                        frequency,
                        token
                    )
                )

        useful_name_tokens.sort(
            key=lambda x: x[0]
        )

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
        # Address tokens
        # ----------------------------------------------------

        address_tokens = get_tokens(
            row.address_norm
        )

        useful_address_tokens = []

        for token in address_tokens:

            frequency = address_freq.get(
                token,
                MAX_TOKEN_FREQUENCY + 1
            )

            if (
                frequency
                <= MAX_TOKEN_FREQUENCY
            ):

                useful_address_tokens.append(
                    (
                        frequency,
                        token
                    )
                )

        useful_address_tokens.sort(
            key=lambda x: x[0]
        )

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
        # Score candidates
        # ----------------------------------------------------

        scored_candidates = []

        for target_id in candidates:

            target_row = target_lookup.get(
                target_id
            )

            if target_row is None:
                continue

            # Country blocking.
            if (
                row.country
                != target_row["country"]
            ):
                continue

            score = score_candidate(
                row,
                target_row,
                name_freq,
                address_freq
            )

            scored_candidates.append(
                (
                    score,
                    target_id
                )
            )

        # ----------------------------------------------------
        # Keep top candidates
        # ----------------------------------------------------

        scored_candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        ranked_candidates = (
            scored_candidates[
                :MAX_CANDIDATES_PER_S1
            ]
        )

        # ----------------------------------------------------
        # Save candidate pairs
        # ----------------------------------------------------

        for _, target_id in ranked_candidates:

            output_rows.append(
                (
                    row.entity_id,
                    target_id
                )
            )

        candidate_count = len(
            ranked_candidates
        )

        total_candidates += (
            candidate_count
        )

        if candidate_count > max_candidates:

            max_candidates = (
                candidate_count
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % PRINT_EVERY == 0:

            mean_candidates = (
                total_candidates
                /
                i
            )

            print(
                f"Processed "
                f"{i:,}/"
                f"{len(s1):,} | "
                f"Mean candidates/S1: "
                f"{mean_candidates:,.1f} | "
                f"Current: "
                f"{candidate_count:,} | "
                f"Max: "
                f"{max_candidates:,}"
            )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    print()
    print("Creating output dataframe...")

    result = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    print(
        "Writing:",
        OUTPUT_FILE
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