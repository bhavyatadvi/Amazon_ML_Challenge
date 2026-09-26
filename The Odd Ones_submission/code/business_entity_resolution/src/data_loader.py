import pandas as pd

from config import TRAIN_DIR, TEST_DIR


def load_training_data():

    source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    return source1, source2, source3, ground_truth


def load_test_data():

    source1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t"
    )

    return source1, source2, source3