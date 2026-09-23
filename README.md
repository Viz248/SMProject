# BRFSS Stroke Prediction Project

This repository contains the data-processing, feature-engineering, model-training, and evaluation pipeline for a BRFSS-based stroke prediction project.

## Repository contents

The project uses Python scripts in the repository root:

- `preprocessing.py` – loads the BRFSS XPT file, cleans it, and writes intermediate CSVs
- `feature_extraction.py` – derives engineered features
- `feature_selection.py` – performs feature filtering and split generation
- `feature_scaling.py` – standardizes the training/test feature matrices
- `classification.py` – trains and evaluates the models and saves predictions/probabilities
- `threshold_tuning.py` – tests decision thresholds for model comparison
- `visualizations.py` – plots model metrics and confusion matrices

## Important: dataset is not committed

This project expects the raw BRFSS dataset to be obtained separately from the team and placed in the repository root.

The raw dataset file should be:

- `LLCP2024.XPT`

Do not commit the dataset, intermediate CSVs, or generated model outputs. They are excluded via `.gitignore`.

## Setup

Clone the repository and open a terminal in the project root.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data acquisition

1. Obtain the BRFSS dataset ZIP from the project team.
2. Extract the contents into the project root directory.
3. Confirm that the file exists at:

```text
<repo-root>/LLCP2024.XPT
```

## Run the pipeline

Run the scripts in this order from the repository root:

```bash
python preprocessing.py
python feature_extraction.py
python feature_selection.py
python feature_scaling.py
python classification.py
python threshold_tuning.py
python visualizations.py
```

The scripts use relative paths and expect their input files to live in the same project folder as the scripts.

## Notes

- Generated outputs such as `.csv`, `.npy`, and model artifacts are intentionally ignored by Git.
- The project should be run from the repository root so the relative file paths resolve correctly.
- If you are working in a different folder, use the repository root as your working directory before executing any script.
