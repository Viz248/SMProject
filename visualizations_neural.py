
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# VISUALIZATIONS - BRFSS STROKE PROJECT
# Uses predictions produced from the neural-feature pipeline.
# ============================================================

y_test = np.load("y_test.npy")

predictions = {
    "Logistic Regression": np.load("y_pred_lr.npy"),
    "Decision Tree": np.load("y_pred_dt.npy"),
    "Random Forest": np.load("y_pred_rf.npy"),
    "Gradient Boosting": np.load("y_pred_gb.npy"),
}

probabilities = {
    "Logistic Regression": np.load("y_prob_lr.npy"),
    "Decision Tree": np.load("y_prob_dt.npy"),
    "Random Forest": np.load("y_prob_rf.npy"),
    "Gradient Boosting": np.load("y_prob_gb.npy"),
}

print("=" * 70)
print("BRFSS STROKE VISUALIZATIONS - NEURAL FEATURES")
print("=" * 70)


# ============================================================
# 1. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

for ax, (model_name, y_pred) in zip(
    axes.ravel(),
    predictions.items(),
):

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle(
    "Confusion Matrix Comparison - Neural Features",
    fontsize=16,
)

plt.tight_layout()
plt.show()


# ============================================================
# 2. ROC CURVE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 7),
)

for model_name, y_prob in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob,
    )

    auc_value = roc_auc_score(
        y_test,
        y_prob,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_value:.4f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Neural Features")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 3. PRECISION-RECALL CURVE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 7),
)

for model_name, y_prob in probabilities.items():

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob,
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison - Neural Features")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. COMPUTE CURRENT PERFORMANCE DYNAMICALLY
# ============================================================
# Unlike the old visualization script, these values are calculated
# from the current neural-feature predictions rather than hard-coded.

results = []

for model_name in predictions:

    y_pred = predictions[model_name]
    y_prob = probabilities[model_name]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob,
        ),
        "PR-AUC": average_precision_score(
            y_test,
            y_prob,
        ),
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CURRENT MODEL PERFORMANCE")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 5. PERFORMANCE PLOT
# ============================================================

results_df.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(14, 7),
)

plt.title("Model Performance Comparison - Neural Features")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend()
plt.grid(
    axis="y",
    alpha=0.3,
)
plt.tight_layout()
plt.show()

print("\nVisualizations generated successfully.")
