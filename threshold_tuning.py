import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# THRESHOLD TUNING - BRFSS STROKE PROJECT
# ============================================================

X_TRAIN_SCALED = "X_train_scaled.csv"
X_TEST_SCALED = "X_test_scaled.csv"

X_TRAIN_FINAL = "X_train_final.csv"
X_TEST_FINAL = "X_test_final.csv"

Y_TRAIN = "y_train_final.csv"
Y_TEST = "y_test_final.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)

X_train_scaled = pd.read_csv(X_TRAIN_SCALED)
X_test_scaled = pd.read_csv(X_TEST_SCALED)

X_train = pd.read_csv(X_TRAIN_FINAL)
X_test = pd.read_csv(X_TEST_FINAL)

y_train = pd.read_csv(Y_TRAIN).squeeze()
y_test = pd.read_csv(Y_TEST).squeeze()


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
# 3. THRESHOLDS TO TEST
# ============================================================

thresholds = [
    0.50,
    0.40,
    0.30,
    0.20,
    0.15,
    0.10,
    0.08,
    0.05,
    0.03,
    0.01
]


all_results = []


# ============================================================
# 4. TRAIN MODELS
# ============================================================

for model_name, (model, train_data, test_data) in models.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    model.fit(
        train_data,
        y_train
    )

    # Get probability of stroke
    y_prob = model.predict_proba(
        test_data
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )

    print("\nThreshold Results:")
    print("-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Accuracy':<12}"
        f"{'Specificity':<12}"
    )


    # ========================================================
    # TEST EACH THRESHOLD
    # ========================================================

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)


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

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        # Confusion matrix

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()


        # Specificity

        specificity = tn / (tn + fp)


        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{accuracy:<12.4f}"
            f"{specificity:<12.4f}"
        )


        all_results.append({

            "Model": model_name,

            "Threshold": threshold,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "Accuracy": accuracy,

            "Specificity": specificity,

            "ROC_AUC": roc_auc,

            "PR_AUC": pr_auc,

            "TN": tn,

            "FP": fp,

            "FN": fn,

            "TP": tp
        })


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "threshold_results.csv",
    index=False
)


# ============================================================
# 6. BEST F1 THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("BEST F1 THRESHOLDS")
print("=" * 70)

for model_name in models.keys():

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    best = model_results.loc[
        model_results["F1"].idxmax()
    ]

    print(
        f"\n{model_name}"
    )

    print(
        f"Threshold : {best['Threshold']:.2f}"
    )

    print(
        f"Precision : {best['Precision']:.4f}"
    )

    print(
        f"Recall    : {best['Recall']:.4f}"
    )

    print(
        f"F1        : {best['F1']:.4f}"
    )

    print(
        f"Accuracy  : {best['Accuracy']:.4f}"
    )

    print(
        f"Specificity: {best['Specificity']:.4f}"
    )


# ============================================================
# 7. HIGH-RECALL OPERATING POINT
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RECALL OPERATING POINT")
print("=" * 70)

for model_name in models.keys():

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    # Find thresholds achieving at least 80% recall

    high_recall = model_results[
        model_results["Recall"] >= 0.80
    ]

    if len(high_recall) > 0:

        # Among these, choose highest F1

        best = high_recall.loc[
            high_recall["F1"].idxmax()
        ]

        print(
            f"\n{model_name}"
        )

        print(
            f"Threshold : {best['Threshold']:.2f}"
        )

        print(
            f"Precision : {best['Precision']:.4f}"
        )

        print(
            f"Recall    : {best['Recall']:.4f}"
        )

        print(
            f"F1        : {best['F1']:.4f}"
        )

        print(
            f"Specificity: {best['Specificity']:.4f}"
        )

    else:

        print(
            f"\n{model_name}: "
            "No threshold achieved 80% recall."
        )


print("\n" + "=" * 70)
print("THRESHOLD TUNING COMPLETE")
print("=" * 70)

print("\nGenerated file:")
print("  threshold_results.csv")