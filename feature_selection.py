import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# FINAL FEATURE SELECTION - BRFSS STROKE PROJECT
# ============================================================

INPUT_FILE = "brfss_extracted.csv"

TRAIN_X_FILE = "X_train_final.csv"
TEST_X_FILE = "X_test_final.csv"
TRAIN_Y_FILE = "y_train_final.csv"
TEST_Y_FILE = "y_test_final.csv"
RANKING_FILE = "final_feature_ranking.csv"


# Number of final features
N_FEATURES = 15

RANDOM_STATE = 42


# ============================================================
# 1. LOAD EXTRACTED DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("FINAL FEATURE SELECTION PIPELINE")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["stroke"])
y = df["stroke"]


# ============================================================
# 3. REMOVE REDUNDANT REPRESENTATIONS
# ============================================================
#
# These derived variables directly encode information already
# represented by the original BRFSS variables.
#
# We retain the original BRFSS representation and remove the
# duplicate representation.
#
# This does NOT mean the underlying concept is unimportant.
# It only prevents the same information from appearing multiple
# times under different column names.

redundant_features = [
    "age_group",
    "heart_attack_history",
    "coronary_history",
    "diabetes",
    "current_smoker",
    "former_smoker",
    "ever_smoked",
    "physically_active",
    "physically_inactive"
]

redundant_features = [
    feature
    for feature in redundant_features
    if feature in X.columns
]

print("\nRedundant representations removed:")

for feature in redundant_features:
    print("  -", feature)

X = X.drop(
    columns=redundant_features
)


# ============================================================
# 4. CHECK REMAINING FEATURES
# ============================================================

print("\nRemaining Candidate Features:")
print(len(X.columns))

for feature in X.columns:
    print("  -", feature)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# The test set is separated BEFORE feature selection.
#
# Feature selection will use ONLY X_train and y_train.
#
# The test set will remain untouched until classification.

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# ============================================================
# 6. MUTUAL INFORMATION
# ============================================================
#
# Mutual Information is calculated ONLY on the training data.

print("\n" + "=" * 70)
print("MUTUAL INFORMATION - TRAINING DATA ONLY")
print("=" * 70)

mi_scores = mutual_info_classif(

    X_train,

    y_train,

    random_state=RANDOM_STATE
)


mi_results = pd.DataFrame({

    "Feature": X_train.columns,

    "Mutual_Information": mi_scores

})


mi_results = mi_results.sort_values(

    by="Mutual_Information",

    ascending=False

)


mi_results["MI_Rank"] = (

    mi_results[
        "Mutual_Information"
    ]
    .rank(
        ascending=False,
        method="min"
    )

)


print(
    mi_results.to_string(
        index=False
    )
)


# ============================================================
# 7. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
#
# Again, ONLY training data is used.

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


rf = RandomForestClassifier(

    n_estimators=200,

    random_state=RANDOM_STATE,

    class_weight="balanced",

    n_jobs=-1,

    max_depth=12
)


rf.fit(
    X_train,
    y_train
)


rf_results = pd.DataFrame({

    "Feature": X_train.columns,

    "RF_Importance": rf.feature_importances_

})


rf_results = rf_results.sort_values(

    by="RF_Importance",

    ascending=False

)


rf_results["RF_Rank"] = (

    rf_results[
        "RF_Importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )

)


print(
    rf_results.to_string(
        index=False
    )
)


# ============================================================
# 8. COMBINE FEATURE RANKINGS
# ============================================================

print("\n" + "=" * 70)
print("COMBINED FEATURE RANKING")
print("=" * 70)


ranking = mi_results[
    [
        "Feature",
        "Mutual_Information",
        "MI_Rank"
    ]
].merge(

    rf_results[
        [
            "Feature",
            "RF_Importance",
            "RF_Rank"
        ]
    ],

    on="Feature"

)


# Average rank between MI and Random Forest.

ranking["Average_Rank"] = (

    ranking["MI_Rank"] +
    ranking["RF_Rank"]

) / 2


ranking = ranking.sort_values(

    by="Average_Rank"

)


print(
    ranking.to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT FINAL FEATURES
# ============================================================

selected_features = (

    ranking

    .head(N_FEATURES)

    ["Feature"]

    .tolist()

)


print("\n" + "=" * 70)
print("FINAL SELECTED FEATURES")
print("=" * 70)


for i, feature in enumerate(

    selected_features,

    start=1

):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 10. APPLY SELECTED FEATURES
# ============================================================
#
# The feature list was determined ONLY from the training data.
#
# We now apply the same feature list to both datasets.

X_train_final = X_train[
    selected_features
].copy()


X_test_final = X_test[
    selected_features
].copy()


# ============================================================
# 11. SAVE FINAL DATASETS
# ============================================================

X_train_final.to_csv(

    TRAIN_X_FILE,

    index=False

)


X_test_final.to_csv(

    TEST_X_FILE,

    index=False

)


y_train.to_csv(

    TRAIN_Y_FILE,

    index=False

)


y_test.to_csv(

    TEST_Y_FILE,

    index=False

)


ranking.to_csv(

    RANKING_FILE,

    index=False

)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE SELECTION COMPLETE")
print("=" * 70)


print("\nFinal Training Features:")
print(X_train_final.shape)


print("\nFinal Testing Features:")
print(X_test_final.shape)


print("\nSelected Features:")

for feature in selected_features:
    print("  -", feature)


print("\nGenerated Files:")

print("  X_train_final.csv")
print("  X_test_final.csv")
print("  y_train_final.csv")
print("  y_test_final.csv")
print("  final_feature_ranking.csv")


print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)