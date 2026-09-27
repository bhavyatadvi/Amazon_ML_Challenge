import os
import re
from collections import Counter, defaultdict

import pandas as pd

from data_loader import load_test_data


# ============================================================
# PRODUCTION CANDIDATE GENERATOR V2
# ============================================================
#
# Goal:
#   Generate output/candidate_pairs.tsv for the test data.
#
# Design:
#   - country-aware blocking
#   - exact normalized name/address protection
#   - rare-token blocking
#   - bounded candidate set per S1
#   - chunked output writing (no giant output_rows list)
#
# This version is intended to be a drop-in replacement for the
# slower production_candidate_generator.py.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


# ------------------------------------------------------------
# Blocking configuration
# ------------------------------------------------------------

# Tokens occurring more often than this are ignored for normal
# token blocking because they create very large candidate sets.
MAX_TOKEN_FREQUENCY = 500

# Number of rarest useful tokens taken from each field.
RAREST_NAME_TOKENS = 3
RAREST_ADDRESS_TOKENS = 3

# Hard upper bound for candidates retained per S1.
MAX_CANDIDATES_PER_S1 = 500

# Write results every N S1 rows.
WRITE_CHUNK_SIZE = 5000

# Progress display.
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
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
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
# TARGET PREPARATION
# ============================================================

def prepare_target(target):
    print("Normalizing target data...")

    target = target.copy()

    target["name_norm"] = (
        target["business_name"]
        .map(normalize_text)
    )

    target["address_norm"] = (
        target["business_address"]
        .map(normalize_text)
    )

    return target


# ============================================================
# INDEX BUILDING
# ============================================================

def build_token_index(target, column):
    """
    Build:

        token -> [entity_id, ...]

    and token frequency.

    The country is not part of the dictionary key here because
    country filtering is applied after candidate retrieval.
    """

    print()
    print(f"Building {column} token index...")

    frequencies = Counter()

    # Pass 1: frequencies.
    for value in target[column]:
        tokens = get_tokens(value)

        for token in tokens:
            frequencies[token] += 1

    print(
        f"Unique {column} tokens: "
        f"{len(frequencies):,}"
    )

    # Pass 2: index only useful-frequency tokens.
    index = defaultdict(list)

    for row in target.itertuples(index=False):

        value = getattr(row, column)

        for token in get_tokens(value):

            if frequencies[token] <= MAX_TOKEN_FREQUENCY:
                index[token].append(row.entity_id)

    print(
        f"{column} index keys: "
        f"{len(index):,}"
    )

    return index, frequencies


# ============================================================
# EXACT LOOKUPS
# ============================================================

def build_exact_lookup(target, column):
    """
    Maps normalized field value -> list of entity IDs.

    Empty values are ignored.
    """

    lookup = defaultdict(list)

    for row in target.itertuples(index=False):

        value = getattr(row, column)

        if value:
            lookup[value].append(row.entity_id)

    return lookup


# ============================================================
# TARGET METADATA
# ============================================================

def build_target_metadata(target):
    """
    Compact lookup:

        entity_id -> (country, name_norm, address_norm)
    """

    metadata = {}

    for row in target.itertuples(index=False):

        metadata[row.entity_id] = (
            row.country,
            row.name_norm,
            row.address_norm,
        )

    return metadata


# ============================================================
# CANDIDATE SCORING
# ============================================================

def candidate_score(
    s1_row,
    target_id,
    target_metadata,
    name_freq,
    address_freq,
):
    """
    Lightweight score used only after blocking.

    Higher is better.

    Exact normalized fields dominate.
    Address overlap receives slightly more weight.
    Rare shared tokens receive additional weight.
    """

    metadata = target_metadata.get(target_id)

    if metadata is None:
        return None

    target_country, target_name, target_address = metadata

    # Country is a hard blocking condition.
    if target_country != s1_row.country:
        return None

    name1 = get_tokens(s1_row.name_norm)
    name2 = get_tokens(target_name)

    address1 = get_tokens(s1_row.address_norm)
    address2 = get_tokens(target_address)

    shared_name = name1.intersection(name2)
    shared_address = address1.intersection(address2)

    exact_name = (
        bool(s1_row.name_norm)
        and s1_row.name_norm == target_name
    )

    exact_address = (
        bool(s1_row.address_norm)
        and s1_row.address_norm == target_address
    )

    score = 0.0

    # Exact fields get a very large advantage.
    if exact_name:
        score += 1_000_000.0

    if exact_address:
        score += 1_000_000.0

    # Shared-token evidence.
    score += len(shared_name) * 100.0
    score += len(shared_address) * 150.0

    # Rare-token evidence.
    for token in shared_name:
        frequency = name_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1,
        )

        if frequency > 0:
            score += 10000.0 / frequency

    for token in shared_address:
        frequency = address_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1,
        )

        if frequency > 0:
            score += 10000.0 / frequency

    return score


# ============================================================
# GENERATE CANDIDATES FOR ONE S1
# ============================================================

def generate_candidates(
    s1_row,
    name_index,
    address_index,
    name_freq,
    address_freq,
    exact_name_lookup,
    exact_address_lookup,
    target_metadata,
):
    """
    Candidate generation order:

    1. exact normalized name
    2. exact normalized address
    3. rare name tokens
    4. rare address tokens

    Candidate IDs are deduplicated.

    Exact matches are protected from the candidate cap.
    """

    candidates = set()

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if s1_row.name_norm:
        for entity_id in exact_name_lookup.get(
            s1_row.name_norm,
            (),
        ):
            metadata = target_metadata.get(entity_id)

            if metadata is not None:
                if metadata[0] == s1_row.country:
                    candidates.add(entity_id)

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    if s1_row.address_norm:
        for entity_id in exact_address_lookup.get(
            s1_row.address_norm,
            (),
        ):
            metadata = target_metadata.get(entity_id)

            if metadata is not None:
                if metadata[0] == s1_row.country:
                    candidates.add(entity_id)

    # Keep exact candidates protected.
    protected = set(candidates)

    # --------------------------------------------------------
    # Rarest useful name tokens
    # --------------------------------------------------------

    useful_name_tokens = []

    for token in get_tokens(s1_row.name_norm):

        frequency = name_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1,
        )

        if frequency <= MAX_TOKEN_FREQUENCY:
            useful_name_tokens.append(
                (frequency, token)
            )

    useful_name_tokens.sort(
        key=lambda x: x[0]
    )

    for _, token in useful_name_tokens[:RAREST_NAME_TOKENS]:

        for entity_id in name_index.get(
            token,
            (),
        ):
            candidates.add(entity_id)

    # --------------------------------------------------------
    # Rarest useful address tokens
    # --------------------------------------------------------

    useful_address_tokens = []

    for token in get_tokens(s1_row.address_norm):

        frequency = address_freq.get(
            token,
            MAX_TOKEN_FREQUENCY + 1,
        )

        if frequency <= MAX_TOKEN_FREQUENCY:
            useful_address_tokens.append(
                (frequency, token)
            )

    useful_address_tokens.sort(
        key=lambda x: x[0]
    )

    for _, token in useful_address_tokens[:RAREST_ADDRESS_TOKENS]:

        for entity_id in address_index.get(
            token,
            (),
        ):
            candidates.add(entity_id)

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []

    for entity_id in candidates:

        score = candidate_score(
            s1_row,
            entity_id,
            target_metadata,
            name_freq,
            address_freq,
        )

        if score is not None:
            scored.append(
                (
                    score,
                    entity_id,
                )
            )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    # Keep protected exact matches even if the normal candidate
    # pool is larger than the cap.
    protected_scored = [
        item
        for item in scored
        if item[1] in protected
    ]

    normal_scored = [
        item
        for item in scored
        if item[1] not in protected
    ]

    remaining_slots = max(
        0,
        MAX_CANDIDATES_PER_S1
        - len(protected_scored),
    )

    ranked = (
        protected_scored
        + normal_scored[:remaining_slots]
    )

    return ranked


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRODUCTION CANDIDATE GENERATOR V2")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print()
    print("Loading TEST data...")

    s1, s2, s3 = load_test_data()

    print(
        f"S1 records: {len(s1):,}"
    )

    print(
        f"S2 records: {len(s2):,}"
    )

    print(
        f"S3 records: {len(s3):,}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    del s2
    del s3

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print()
    print("Preparing S1...")

    s1 = s1.copy()

    s1["name_norm"] = (
        s1["business_name"]
        .map(normalize_text)
    )

    s1["address_norm"] = (
        s1["business_address"]
        .map(normalize_text)
    )

    target = prepare_target(target)

    # --------------------------------------------------------
    # Target metadata
    # --------------------------------------------------------

    print()
    print("Building target metadata...")

    target_metadata = build_target_metadata(
        target
    )

    # --------------------------------------------------------
    # Exact lookups
    # --------------------------------------------------------

    print()
    print("Building exact normalized lookups...")

    exact_name_lookup = build_exact_lookup(
        target,
        "name_norm",
    )

    exact_address_lookup = build_exact_lookup(
        target,
        "address_norm",
    )

    print(
        "Exact name keys:",
        f"{len(exact_name_lookup):,}",
    )

    print(
        "Exact address keys:",
        f"{len(exact_address_lookup):,}",
    )

    # --------------------------------------------------------
    # Token indexes
    # --------------------------------------------------------

    name_index, name_freq = build_token_index(
        target,
        "name_norm",
    )

    address_index, address_freq = build_token_index(
        target,
        "address_norm",
    )

    # target dataframe is no longer required after the indexes
    # and metadata are built.
    del target

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    print()
    print("Preparing output...")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    first_write = True

    chunk_rows = []

    total_candidates = 0
    max_candidates = 0

    processed = 0

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print("Generating candidates...")
    print(
        "Maximum candidates/S1:",
        MAX_CANDIDATES_PER_S1,
    )

    for row in s1.itertuples(
        index=False
    ):

        ranked_candidates = generate_candidates(
            row,
            name_index,
            address_index,
            name_freq,
            address_freq,
            exact_name_lookup,
            exact_address_lookup,
            target_metadata,
        )

        candidate_count = len(
            ranked_candidates
        )

        total_candidates += candidate_count

        if candidate_count > max_candidates:
            max_candidates = candidate_count

        for _, target_id in ranked_candidates:

            chunk_rows.append(
                (
                    row.entity_id,
                    target_id,
                )
            )

        processed += 1

        # ----------------------------------------------------
        # Chunk write
        # ----------------------------------------------------

        if (
            len(chunk_rows)
            >= WRITE_CHUNK_SIZE
        ):

            chunk_df = pd.DataFrame(
                chunk_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
            )

            chunk_df.to_csv(
                OUTPUT_FILE,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False
            chunk_rows.clear()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % PRINT_EVERY == 0:

            mean_candidates = (
                total_candidates
                / processed
            )

            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(s1):,} | "
                f"Mean candidates/S1: "
                f"{mean_candidates:,.1f} | "
                f"Current: "
                f"{candidate_count:,} | "
                f"Max: "
                f"{max_candidates:,}"
            )

    # --------------------------------------------------------
    # Final chunk
    # --------------------------------------------------------

    if chunk_rows:

        chunk_df = pd.DataFrame(
            chunk_rows,
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
        )

        chunk_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        chunk_rows.clear()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    mean_candidates = (
        total_candidates
        / len(s1)
        if len(s1)
        else 0
    )

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Candidate pairs:",
        f"{total_candidates:,}",
    )

    print(
        "Mean candidates/S1:",
        f"{mean_candidates:,.2f}",
    )

    print(
        "Maximum candidates/S1:",
        f"{max_candidates:,}",
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
