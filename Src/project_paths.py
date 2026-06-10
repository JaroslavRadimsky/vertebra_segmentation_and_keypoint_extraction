from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
ATLAS_DATA_DIR = DATA_DIR / "atlas_vertebra"
ATLAS_FOLDS_DIR = DATA_DIR / "folds" / "atlas_vertebra"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SEGMENTATION_CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints" / "segmentation"
KEYPOINT_CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints" / "keypoint"

OUTPUTS_DIR = REPO_ROOT / "outputs"
ANALYSES_DIR = OUTPUTS_DIR / "analyses"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
RESULTS_DIR = OUTPUTS_DIR / "results"
RUNS_DIR = OUTPUTS_DIR / "runs"
SEQUENCES_DIR = OUTPUTS_DIR / "sequences"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
