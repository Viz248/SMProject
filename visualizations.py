import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# VISUALIZATIONS - BRFSS STROKE PROJECT
# ============================================================


# ============================================================
# 1. LOAD SAVED RESULTS
# ============================================================

y_test = np.load("y_test.npy")

y_pred_lr = np.load("y_pred_lr.npy")
y_pred_dt = np.load("y_pred_dt.npy")
y_pred_rf = np.load("y_pred_rf.npy")
y_pred_gb = np.load("y_pred_gb.npy")

y_prob_lr = np.load("y_prob_lr.npy")
y_prob_dt = np.load("y_prob_dt.npy")
y_prob_rf = np.load("y_prob_rf.npy")
y_prob_gb = np.load("y_prob_gb.npy")


print("=" * 70)
print("BRFSS STROKE VISUALIZATIONS")
print("=" * 70)


# ============================================================
# 2. STORE PREDICTIONS
# ============================================================

predictions = {

    "Logistic Regression": y_pred_lr,

    "Decision Tree": y_pred_dt,

    "Random Forest": y_pred_rf,

    "Gradient Boosting": y_pred_gb

}


# ============================================================
# 3. STORE PROBABILITIES
# ============================================================

probabilities = {

    "Logistic Regression": y_prob_lr,

    "Decision Tree": y_prob_dt,

    "Random Forest": y_prob_rf,

    "Gradient Boosting": y_prob_gb

}


# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)


for ax, (model_name, y_pred) in zip(
    axes.ravel(),
    predictions.items()
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"]
    )


    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )


plt.suptitle(
    "Confusion Matrix Comparison",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 5. ROC CURVE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 7)
)


for model_name, y_prob in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )


    auc = roc_auc_score(
        y_test,
        y_prob
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.4f})"
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 6. PRECISION-RECALL CURVE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 7)
)


for model_name, y_prob in probabilities.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )


    pr_auc = average_precision_score(
        y_test,
        y_prob
    )


    plt.plot(
        recall,
        precision,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})"
    )


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. MODEL PERFORMANCE COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Decision Tree",

        "Random Forest",

        "Gradient Boosting"

    ],

    "Accuracy": [

        0.7314,

        0.7016,

        0.7269,

        0.9547

    ],

    "Precision": [

        0.1143,

        0.1065,

        0.1130,

        0.6667

    ],

    "Recall": [

        0.7306,

        0.7561,

        0.7348,

        0.0005

    ],

    "F1": [

        0.1977,

        0.1867,

        0.1959,

        0.0010

    ],

    "ROC-AUC": [

        0.8116,

        0.7996,

        0.8099,

        0.8134

    ],

    "PR-AUC": [

        0.1756,

        0.1676,

        0.1763,

        0.1817

    ]

})


results.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(14, 7)
)


plt.title(
    "Model Performance Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    rotation=0
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


print("\nVisualizations generated successfully.")