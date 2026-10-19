import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD BRFSS 2024 DATASET
# ============================================================

# Download LLCP2024.XPT from the official CDC BRFSS website
# and place it in the same folder as this script.

FILE_PATH = "LLCP2024.XPT"

df = pd.read_sas(FILE_PATH, format="xport")

print("=" * 70)
print("BRFSS DATASET")
print("=" * 70)

print("\nOriginal Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nNumber of Columns:")
print(len(df.columns))


# ============================================================
# 2. SELECT FEATURES
# ============================================================

# Stroke target:
# CVDSTRK3
#   1 = Yes
#   2 = No
#   7 = Don't know / Not sure
#   9 = Refused
#
# Selected predictor variables:
#
# SEXVAR     -> Sex
# _AGEG5YR   -> Age group
# GENHLTH    -> General health
# PHYSHLTH   -> Number of physically unhealthy days
# MENTHLTH   -> Number of mentally unhealthy days
# EXERANY2   -> Physical activity
# _BMI5      -> BMI
# _SMOKER3   -> Smoking status
# DIABETE4   -> Diabetes
# CVDINFR4   -> Heart attack
# CVDCRHD4   -> Coronary heart disease
# CHCKDNY2   -> Kidney disease
# HAVARTH4   -> Arthritis
# EDUCA      -> Education
# INCOME3    -> Income category


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
    "INCOME3"
]


# Check that all required columns exist

missing_columns = [
    column for column in selected_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Some required BRFSS variables are not present."
    )


df = df[selected_columns].copy()

print("\nSelected Dataset Shape:")
print(df.shape)


# ============================================================
# 3. INITIAL DATA EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("INITIAL DATA EXPLORATION")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStroke Distribution:")
print(df["CVDSTRK3"].value_counts(dropna=False))


# ============================================================
# 4. CLEAN SPECIAL BRFSS VALUES
# ============================================================

# BRFSS uses special response codes such as:
#
# 7 / 9
# 77 / 99
#
# for responses such as "Don't know" and "Refused".
#
# These should NOT be treated as real measurements.


# ----------------------------
# TARGET
# ----------------------------

df["CVDSTRK3"] = df["CVDSTRK3"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# SEX
# ----------------------------

df["SEXVAR"] = df["SEXVAR"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# AGE GROUP
# ----------------------------

df["_AGEG5YR"] = df["_AGEG5YR"].replace({
    14: np.nan
})


# ----------------------------
# GENERAL HEALTH
# ----------------------------

df["GENHLTH"] = df["GENHLTH"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# PHYSICAL HEALTH
# ----------------------------

# 88 = 0 days
# 77 / 99 = Don't know / Refused

df["PHYSHLTH"] = df["PHYSHLTH"].replace({
    88: 0,
    77: np.nan,
    99: np.nan
})


# ----------------------------
# MENTAL HEALTH
# ----------------------------

df["MENTHLTH"] = df["MENTHLTH"].replace({
    88: 0,
    77: np.nan,
    99: np.nan
})


# ----------------------------
# PHYSICAL ACTIVITY
# ----------------------------

df["EXERANY2"] = df["EXERANY2"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# BMI
# ----------------------------

# _BMI5 is stored as BMI × 100.
#
# Example:
# 2500 -> 25.00 BMI

df["_BMI5"] = df["_BMI5"].replace({
    0: np.nan,
    9999: np.nan
})

df["_BMI5"] = df["_BMI5"] / 100


# ----------------------------
# SMOKING
# ----------------------------

df["_SMOKER3"] = df["_SMOKER3"].replace({
    9: np.nan
})


# ----------------------------
# DIABETES
# ----------------------------

df["DIABETE4"] = df["DIABETE4"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# HEART ATTACK
# ----------------------------

df["CVDINFR4"] = df["CVDINFR4"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# CORONARY HEART DISEASE
# ----------------------------

df["CVDCRHD4"] = df["CVDCRHD4"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# KIDNEY DISEASE
# ----------------------------

df["CHCKDNY2"] = df["CHCKDNY2"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# ARTHRITIS
# ----------------------------

df["HAVARTH4"] = df["HAVARTH4"].replace({
    7: np.nan,
    9: np.nan
})


# ----------------------------
# EDUCATION
# ----------------------------

df["EDUCA"] = df["EDUCA"].replace({
    9: np.nan
})


# ----------------------------
# INCOME
# ----------------------------

df["INCOME3"] = df["INCOME3"].replace({
    77: np.nan,
    99: np.nan
})


# ============================================================
# 5. REMOVE ROWS WITH UNKNOWN STROKE LABEL
# ============================================================

# We cannot train a supervised classifier if the target
# variable is unknown.

df = df.dropna(subset=["CVDSTRK3"])


# Convert target:
#
# 1 = Stroke
# 2 = No Stroke
#
# Final:
#
# 1 = Stroke
# 0 = No Stroke

df["stroke"] = df["CVDSTRK3"].map({
    1: 1,
    2: 0
})

df.drop("CVDSTRK3", axis=1, inplace=True)


# ============================================================
# 6. HANDLE REMAINING MISSING VALUES
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
    "INCOME3"
]

numerical_columns = [
    "PHYSHLTH",
    "MENTHLTH",
    "_BMI5"
]


# Fill categorical variables using mode

for column in categorical_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# Fill numerical variables using median

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ============================================================
# 7. CHECK CLEANED DATA
# ============================================================

print("\n" + "=" * 70)
print("AFTER CLEANING")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nStroke Distribution:")
print(df["stroke"].value_counts())

print("\nStroke Percentage:")
print(
    df["stroke"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("stroke", axis=1)

y = df["stroke"]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

df.to_csv("brfss_cleaned.csv", index=False)

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

# Stratify ensures the proportion of stroke/non-stroke cases
# remains approximately the same in train and test sets.

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining Set:")
print(X_train.shape)

print("\nTesting Set:")
print(X_test.shape)


# ============================================================
# 10. FEATURE SCALING
# ============================================================

# Scale numerical features.
#
# IMPORTANT:
# Fit the scaler ONLY on training data.
# This prevents data leakage.

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)


# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

X_train.to_csv(
    "X_train_brfss.csv",
    index=False
)

X_test.to_csv(
    "X_test_brfss.csv",
    index=False
)

y_train.to_csv(
    "y_train_brfss.csv",
    index=False
)

y_test.to_csv(
    "y_test_brfss.csv",
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nTraining features:", X_train.shape)
print("Testing features :", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

print("\nGenerated files:")
print("  X_train_brfss.csv")
print("  X_test_brfss.csv")
print("  y_train_brfss.csv")
print("  y_test_brfss.csv")