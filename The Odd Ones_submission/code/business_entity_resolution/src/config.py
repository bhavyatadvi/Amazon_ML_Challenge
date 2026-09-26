from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# src/
SRC_DIR = Path(__file__).resolve().parent

# business_entity_resolution/
PROJECT_DIR = SRC_DIR.parent

# code/
CODE_DIR = PROJECT_DIR.parent

# team_submission/
SUBMISSION_DIR = CODE_DIR.parent

# Original challenge resource
CHALLENGE_ROOT = SUBMISSION_DIR.parent / "student_resource"

# Dataset directories
DATASET_DIR = CHALLENGE_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Submission output directory
OUTPUT_DIR = SUBMISSION_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)