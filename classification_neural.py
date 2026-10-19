
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CLASSIFICATION - BRFSS STROKE PROJECT
# Uses learned neural features instead of handcrafted selected
# features.
# ============================================================

X_TRAIN_FILE = "X_train_neural.csv"
X_TEST_FILE = "X_test_neural.csv"

Y_TRAIN_FILE = "y_train_brfss.csv"
Y_TEST_FILE = "y_test_brfss.csv"

RESULTS_FILE = "classification_results_neural.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("BRFSS STROKE CLASSIFICATION - NEURAL FEATURES")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN_FILE)
X_test = pd.read_csv(X_TEST_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze("columns")
y_test = pd.read_csv(Y_TEST_FILE).squeeze("columns")

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))
print("Learned Features:", X_train.shape[1])

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ============================================================
# 2. DEFINE DOWNSTREAM MODELS
# ============================================================
# The classifiers are unchanged from the original project.
# Only their input representation has changed.

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


# ============================================================
# 3. TRAIN AND EVALUATE
# ============================================================

results = []
predictions = {}
probabilities = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    predictions[model_name] = y_pred
    probabilities[model_name] = y_prob

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )
    pr_auc = average_precision_score(
        y_test,
        y_prob,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nResults:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
    })


# ============================================================
# 4. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON - NEURAL FEATURES")
print("=" * 70)

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "PR_AUC",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 5. BEST METRIC VALUES
# ============================================================
# These are reported as metric maxima only; no overall model
# ranking is imposed.

for metric, label in [
    ("Recall", "Highest Recall"),
    ("F1_Score", "Highest F1"),
    ("ROC_AUC", "Highest ROC-AUC"),
    ("PR_AUC", "Highest PR-AUC"),
]:
    row = results_df.loc[results_df[metric].idxmax()]

    print(
        f"\n{label}: "
        f"{row['Model']} ({row[metric]:.4f})"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

np.save("y_test.npy", y_test)

np.save(
    "y_pred_lr.npy",
    predictions["Logistic Regression"],
)
np.save(
    "y_pred_dt.npy",
    predictions["Decision Tree"],
)
np.save(
    "y_pred_rf.npy",
    predictions["Random Forest"],
)
np.save(
    "y_pred_gb.npy",
    predictions["Gradient Boosting"],
)

np.save(
    "y_prob_lr.npy",
    probabilities["Logistic Regression"],
)
np.save(
    "y_prob_dt.npy",
    probabilities["Decision Tree"],
)
np.save(
    "y_prob_rf.npy",
    probabilities["Random Forest"],
)
np.save(
    "y_prob_gb.npy",
    probabilities["Gradient Boosting"],
)

print("\n" + "=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)
print("\nGenerated file:")
print("  classification_results_neural.csv")
