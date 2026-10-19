import pandas as pd
import numpy as np

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
    confusion_matrix
)


# ============================================================
# CLASSIFICATION - BRFSS STROKE PROJECT
# ============================================================

# Files created by the previous stages

X_TRAIN_SCALED = "X_train_scaled.csv"
X_TEST_SCALED = "X_test_scaled.csv"

X_TRAIN_FINAL = "X_train_final.csv"
X_TEST_FINAL = "X_test_final.csv"

Y_TRAIN = "y_train_final.csv"
Y_TEST = "y_test_final.csv"

RESULTS_FILE = "classification_results.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("BRFSS STROKE CLASSIFICATION")
print("=" * 70)


# Scaled data → Logistic Regression

X_train_scaled = pd.read_csv(
    X_TRAIN_SCALED
)

X_test_scaled = pd.read_csv(
    X_TEST_SCALED
)


# Unscaled data → Tree-based models

X_train = pd.read_csv(
    X_TRAIN_FINAL
)

X_test = pd.read_csv(
    X_TEST_FINAL
)


y_train = pd.read_csv(
    Y_TRAIN
).squeeze()


y_test = pd.read_csv(
    Y_TEST
).squeeze()


print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))

print("\nNumber of Features:")
print(X_train.shape[1])

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ============================================================
# 2. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": (

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        X_train_scaled,
        X_test_scaled

    ),

    "Decision Tree": (

        DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=20,
            random_state=42
        ),

        X_train,
        X_test

    ),

    "Random Forest": (

        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        ),

        X_train,
        X_test

    ),

    "Gradient Boosting": (

        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        X_train,
        X_test

    )
}


# ============================================================
# 3. TRAIN AND EVALUATE
# ============================================================

results = []


# Store predictions and probabilities for visualizations

predictions = {}
probabilities = {}


for model_name, (model, train_data, test_data) in models.items():

    print("\n" + "=" * 70)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        train_data,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        test_data
    )


    # Store predictions for visualization

    predictions[model_name] = y_pred


    # Probability of positive class

    if hasattr(
        model,
        "predict_proba"
    ):

        y_prob = model.predict_proba(
            test_data
        )[:, 1]

    else:

        y_prob = model.decision_function(
            test_data
        )


    # Store probabilities for visualization

    probabilities[model_name] = y_prob


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

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

        "True_Positives": tp

    })


# ============================================================
# 4. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
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
            "PR_AUC"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 5. BEST MODELS
# ============================================================

print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)


best_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

best_f1 = results_df.loc[
    results_df["F1_Score"].idxmax()
]

best_roc = results_df.loc[
    results_df["ROC_AUC"].idxmax()
]

best_pr = results_df.loc[
    results_df["PR_AUC"].idxmax()
]


print(
    "\nBest Recall:"
)

print(
    f"{best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)


print(
    "\nBest F1:"
)

print(
    f"{best_f1['Model']} "
    f"({best_f1['F1_Score']:.4f})"
)


print(
    "\nBest ROC-AUC:"
)

print(
    f"{best_roc['Model']} "
    f"({best_roc['ROC_AUC']:.4f})"
)


print(
    "\nBest PR-AUC:"
)

print(
    f"{best_pr['Model']} "
    f"({best_pr['PR_AUC']:.4f})"
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 7. SAVE PREDICTIONS FOR VISUALIZATIONS
# ============================================================

np.save(
    "y_test.npy",
    y_test
)


np.save(
    "y_pred_lr.npy",
    predictions["Logistic Regression"]
)

np.save(
    "y_pred_dt.npy",
    predictions["Decision Tree"]
)

np.save(
    "y_pred_rf.npy",
    predictions["Random Forest"]
)

np.save(
    "y_pred_gb.npy",
    predictions["Gradient Boosting"]
)


np.save(
    "y_prob_lr.npy",
    probabilities["Logistic Regression"]
)

np.save(
    "y_prob_dt.npy",
    probabilities["Decision Tree"]
)

np.save(
    "y_prob_rf.npy",
    probabilities["Random Forest"]
)

np.save(
    "y_prob_gb.npy",
    probabilities["Gradient Boosting"]
)


# ============================================================
# 8. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print("  classification_results.csv")
print("  y_test.npy")
print("  y_pred_lr.npy")
print("  y_pred_dt.npy")
print("  y_pred_rf.npy")
print("  y_pred_gb.npy")
print("  y_prob_lr.npy")
print("  y_prob_dt.npy")
print("  y_prob_rf.npy")
print("  y_prob_gb.npy")