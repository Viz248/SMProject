
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PREPROCESSING - BRFSS STROKE PROJECT
# Neural Feature Learning Version
#
# This script:
# 1. Loads BRFSS 2024
# 2. Selects the original predictor variables
# 3. Cleans BRFSS special response codes
# 4. Creates the binary stroke target
# 5. Splits train/test data
# 6. Performs TRAIN-ONLY imputation and scaling
# 7. Saves files for the neural feature-learning stage
#
# This script does NOT:
# - create handcrafted derived features
# - perform Mutual Information selection
# - perform Random Forest feature selection
# ============================================================


FILE_PATH = "LLCP2024.XPT"

TRAIN_X_FILE = "X_train_brfss.csv"
TEST_X_FILE = "X_test_brfss.csv"
TRAIN_Y_FILE = "y_train_brfss.csv"
TEST_Y_FILE = "y_test_brfss.csv"

RANDOM_STATE = 42


# ============================================================
# 1. LOAD BRFSS 2024 DATASET
# ============================================================

print("=" * 70)
print("BRFSS 2024 PREPROCESSING")
print("=" * 70)

df = pd.read_sas(
    FILE_PATH,
    format="xport",
)

print("\nOriginal Shape:")
print(df.shape)

print("\nNumber of Columns:")
print(len(df.columns))


# ============================================================
# 2. SELECT ORIGINAL FEATURES
# ============================================================

# Stroke target:
# CVDSTRK3
#   1 = Yes
#   2 = No
#   7 = Don't know / Not sure
#   9 = Refused

selected_columns = [
    "CVDSTRK3",
    "SEXVAR",
    "_AGEG5YR",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "EXERANY2",
    "_BMI5",
    "_SMOKER3",
    "DIABETE4",
    "CVDINFR4",
    "CVDCRHD4",
    "CHCKDNY2",
    "HAVARTH4",
    "EDUCA",
    "INCOME3",
]

missing_columns = [
    column
    for column in selected_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required BRFSS variables are missing:\n"
        + str(missing_columns)
    )

df = df[selected_columns].copy()

print("\nSelected Dataset Shape:")
print(df.shape)


# ============================================================
# 3. CLEAN SPECIAL BRFSS VALUES
# ============================================================

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

df["CVDSTRK3"] = df["CVDSTRK3"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# SEX
# ------------------------------------------------------------

df["SEXVAR"] = df["SEXVAR"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# AGE GROUP
# ------------------------------------------------------------

df["_AGEG5YR"] = df["_AGEG5YR"].replace({
    14: np.nan,
})


# ------------------------------------------------------------
# GENERAL HEALTH
# ------------------------------------------------------------

df["GENHLTH"] = df["GENHLTH"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# PHYSICAL HEALTH
# ------------------------------------------------------------

# 88 = 0 days
# 77 / 99 = Don't know / Refused

df["PHYSHLTH"] = df["PHYSHLTH"].replace({
    88: 0,
    77: np.nan,
    99: np.nan,
})


# ------------------------------------------------------------
# MENTAL HEALTH
# ------------------------------------------------------------

df["MENTHLTH"] = df["MENTHLTH"].replace({
    88: 0,
    77: np.nan,
    99: np.nan,
})


# ------------------------------------------------------------
# PHYSICAL ACTIVITY
# ------------------------------------------------------------

df["EXERANY2"] = df["EXERANY2"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# BMI
# ------------------------------------------------------------

# _BMI5 is stored as BMI x 100.
# Example:
# 2500 -> 25.00 BMI

df["_BMI5"] = df["_BMI5"].replace({
    0: np.nan,
    9999: np.nan,
})

df["_BMI5"] = df["_BMI5"] / 100.0


# ------------------------------------------------------------
# SMOKING
# ------------------------------------------------------------

df["_SMOKER3"] = df["_SMOKER3"].replace({
    9: np.nan,
})


# ------------------------------------------------------------
# DIABETES
# ------------------------------------------------------------

df["DIABETE4"] = df["DIABETE4"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# HEART ATTACK
# ------------------------------------------------------------

df["CVDINFR4"] = df["CVDINFR4"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# CORONARY HEART DISEASE
# ------------------------------------------------------------

df["CVDCRHD4"] = df["CVDCRHD4"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# KIDNEY DISEASE
# ------------------------------------------------------------

df["CHCKDNY2"] = df["CHCKDNY2"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# ARTHRITIS
# ------------------------------------------------------------

df["HAVARTH4"] = df["HAVARTH4"].replace({
    7: np.nan,
    9: np.nan,
})


# ------------------------------------------------------------
# EDUCATION
# ------------------------------------------------------------

df["EDUCA"] = df["EDUCA"].replace({
    9: np.nan,
})


# ------------------------------------------------------------
# INCOME
# ------------------------------------------------------------

df["INCOME3"] = df["INCOME3"].replace({
    77: np.nan,
    99: np.nan,
})


# ============================================================
# 4. REMOVE UNKNOWN STROKE LABELS
# ============================================================

df = df.dropna(
    subset=["CVDSTRK3"]
).copy()


# Convert:
# 1 = Stroke
# 2 = No Stroke
#
# Final:
# 1 = Stroke
# 0 = No Stroke

df["stroke"] = df["CVDSTRK3"].map({
    1: 1,
    2: 0,
})

df.drop(
    columns=["CVDSTRK3"],
    inplace=True,
)


# Remove any rows for which the mapping unexpectedly failed.
df = df.dropna(
    subset=["stroke"]
).copy()

df["stroke"] = df["stroke"].astype(int)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["stroke"]
)

y = df["stroke"]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(
    y.value_counts(normalize=True)
     .mul(100)
     .round(2)
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================
#
# The split happens BEFORE imputation/scaling so information
# from the test set cannot influence preprocessing parameters.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)


# ============================================================
# 7. DEFINE FEATURE TYPES
# ============================================================

categorical_columns = [
    "SEXVAR",
    "_AGEG5YR",
    "GENHLTH",
    "EXERANY2",
    "_SMOKER3",
    "DIABETE4",
    "CVDINFR4",
    "CVDCRHD4",
    "CHCKDNY2",
    "HAVARTH4",
    "EDUCA",
    "INCOME3",
]

numerical_columns = [
    "PHYSHLTH",
    "MENTHLTH",
    "_BMI5",
]


# ============================================================
# 8. TRAIN-ONLY IMPUTATION
# ============================================================
#
# Categorical variables -> training-set mode
# Numerical variables   -> training-set median
#
# The test set is transformed using parameters learned only
# from X_train.

for column in categorical_columns:

    mode_values = X_train[column].mode()

    if mode_values.empty:
        raise ValueError(
            f"Cannot calculate mode for categorical column: {column}"
        )

    train_mode = mode_values.iloc[0]

    X_train[column] = X_train[column].fillna(train_mode)
    X_test[column] = X_test[column].fillna(train_mode)


for column in numerical_columns:

    train_median = X_train[column].median()

    if pd.isna(train_median):
        raise ValueError(
            f"Cannot calculate median for numerical column: {column}"
        )

    X_train[column] = X_train[column].fillna(train_median)
    X_test[column] = X_test[column].fillna(train_median)


# ============================================================
# 9. TRAIN-ONLY NUMERICAL SCALING
# ============================================================
#
# Categorical variables remain as their original BRFSS codes.
# neural_feature_learning.py one-hot encodes them before the
# neural network is trained.

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)


# ============================================================
# 10. FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING CHECKS")
print("=" * 70)

print("\nRemaining missing values - TRAIN:")
print(
    X_train.isnull().sum()
)

print("\nRemaining missing values - TEST:")
print(
    X_test.isnull().sum()
)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


if X_train.isnull().sum().sum() != 0:
    raise ValueError(
        "Training data still contains missing values."
    )

if X_test.isnull().sum().sum() != 0:
    raise ValueError(
        "Testing data still contains missing values."
    )


# ============================================================
# 11. SAVE DATA FOR NEURAL FEATURE LEARNING
# ============================================================

X_train.to_csv(
    TRAIN_X_FILE,
    index=False,
)

X_test.to_csv(
    TEST_X_FILE,
    index=False,
)

y_train.to_csv(
    TRAIN_Y_FILE,
    index=False,
)

y_test.to_csv(
    TEST_Y_FILE,
    index=False,
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print(f"  {TRAIN_X_FILE}")
print(f"  {TEST_X_FILE}")
print(f"  {TRAIN_Y_FILE}")
print(f"  {TEST_Y_FILE}")

print("\nNext step:")
print("  python neural_feature_learning.py")
