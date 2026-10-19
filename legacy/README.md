# Legacy Pipeline (Preserved Baseline)

This directory preserves the original teammate pipeline scripts for historical attribution, provenance, and baseline reproduction.

## Scripts

1. `preprocessing.py`: Loads `LLCP2024.XPT`, extracts 15 baseline survey predictors + `CVDSTRK3`, cleans codes, imputes missing values, and exports `brfss_cleaned.csv`.
2. `feature_extraction.py`: Derives age group indicators, general health score, interaction features, and exports `brfss_extracted.csv`.
3. `feature_selection.py`: Performs mutual information and Random Forest feature ranking, partitions train/test data, and exports `X_train_final.csv`, `X_test_final.csv`, `y_train_final.csv`, `y_test_final.csv`.
4. `feature_scaling.py`: Standardizes numerical columns using `StandardScaler` fitted on training only, exporting `X_train_scaled.csv` and `X_test_scaled.csv`.
5. `classification.py`: Trains Logistic Regression (on scaled inputs) and Decision Tree, Random Forest, and Gradient Boosting (on unscaled inputs), saving predictions and `classification_results.csv`.
6. `threshold_tuning.py`: Evaluates decision threshold sweeps on probability outputs and saves `threshold_results.csv`.
7. `visualizations.py`: Generates ROC curves, Precision-Recall curves, confusion matrices, and model comparison figures.

Note: The research package (`truststroke/` package and root pipeline) replaces these scripts with a modular, rigorous framework addressing data leakage, probability calibration, design weights, uncertainty assessment (CAS), explanation stability (DSI), and robustness under feature availability shifts (RAI).
