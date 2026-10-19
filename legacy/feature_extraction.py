import pandas as pd
import numpy as np


# ============================================================
# FEATURE EXTRACTION - BRFSS STROKE PROJECT
# ============================================================

INPUT_FILE = "brfss_cleaned.csv"
OUTPUT_FILE = "brfss_extracted.csv"


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("FEATURE EXTRACTION")
print("=" * 70)

print("\nInput Shape:")
print(df.shape)

print("\nInput Columns:")
print(df.columns.tolist())


# ============================================================
# 2. AGE FEATURES
# ============================================================

# _AGEG5YR is an ordered age-group variable.
#
# 1  = 18-24
# 2  = 25-29
# 3  = 30-34
# ...
# 10 = 65-69
# 11 = 70-74
# 12 = 75-79
# 13 = 80+
#
# Keep the original ordered group and derive an older-adult
# indicator.

df["age_group"] = df["_AGEG5YR"]

df["older_adult"] = (
    df["_AGEG5YR"] >= 10
).astype(int)


# ============================================================
# 3. BMI FEATURES
# ============================================================

# _BMI5 was converted to actual BMI during preprocessing.

def bmi_category(bmi):

    if bmi < 18.5:
        return 0

    elif bmi < 25:
        return 1

    elif bmi < 30:
        return 2

    else:
        return 3


df["bmi_category"] = df["_BMI5"].apply(
    bmi_category
)

df["obese"] = (
    df["_BMI5"] >= 30
).astype(int)


# ============================================================
# 4. PHYSICAL + MENTAL HEALTH FEATURES
# ============================================================

# PHYSHLTH = physically unhealthy days
# MENTHLTH = mentally unhealthy days
#
# Combine the two into a transparent derived feature.

df["health_days_burden"] = (
    df["PHYSHLTH"] +
    df["MENTHLTH"]
)

# Indicator for frequent unhealthy days.

df["frequent_unhealthy_days"] = (
    (df["PHYSHLTH"] >= 14) |
    (df["MENTHLTH"] >= 14)
).astype(int)


# ============================================================
# 5. CARDIOVASCULAR FEATURES
# ============================================================

# CVDINFR4:
#   1 = Yes
#   2 = No
#
# CVDCRHD4:
#   1 = Yes
#   2 = No

df["heart_attack_history"] = (
    df["CVDINFR4"] == 1
).astype(int)

df["coronary_history"] = (
    df["CVDCRHD4"] == 1
).astype(int)


# At least one cardiovascular condition.

df["cardiovascular_history"] = (
    (df["heart_attack_history"] == 1) |
    (df["coronary_history"] == 1)
).astype(int)


# ============================================================
# 6. DIABETES FEATURES
# ============================================================

# DIABETE4:
#   1 = Diabetes
#   2 = No
#   3 = Pre-diabetes / borderline
#   4 = No
#
# Create separate indicators.

df["diabetes"] = (
    df["DIABETE4"] == 1
).astype(int)

df["prediabetes"] = (
    df["DIABETE4"] == 3
).astype(int)


# ============================================================
# 7. SMOKING FEATURES
# ============================================================

# _SMOKER3:
#   1 = Current smoker
#   2 = Former smoker
#   3 = Never smoked
#   4 = Unknown

df["current_smoker"] = (
    df["_SMOKER3"] == 1
).astype(int)

df["former_smoker"] = (
    df["_SMOKER3"] == 2
).astype(int)

df["ever_smoked"] = (
    (df["_SMOKER3"] == 1) |
    (df["_SMOKER3"] == 2)
).astype(int)


# ============================================================
# 8. PHYSICAL ACTIVITY FEATURES
# ============================================================

# EXERANY2:
#   1 = Yes
#   2 = No

df["physically_active"] = (
    df["EXERANY2"] == 1
).astype(int)

df["physically_inactive"] = (
    df["EXERANY2"] == 2
).astype(int)


# ============================================================
# 9. GENERAL HEALTH FEATURE
# ============================================================

# GENHLTH:
#   1 = Excellent
#   2 = Very good
#   3 = Good
#   4 = Fair
#   5 = Poor

df["fair_poor_health"] = (
    df["GENHLTH"] >= 4
).astype(int)


# ============================================================
# 10. EDUCATION FEATURE
# ============================================================

# EDUCA:
# Higher values indicate higher educational attainment.
#
# Create a simple college-education indicator.

df["college_education"] = (
    df["EDUCA"] >= 5
).astype(int)


# ============================================================
# 11. EXISTING VARIABLES
# ============================================================

# We retain the original variables as well.
#
# Feature selection later will determine whether the original
# representation or the derived representation is more useful.


original_features = [
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


derived_features = [
    "age_group",
    "older_adult",
    "bmi_category",
    "obese",
    "health_days_burden",
    "frequent_unhealthy_days",
    "heart_attack_history",
    "coronary_history",
    "cardiovascular_history",
    "diabetes",
    "prediabetes",
    "current_smoker",
    "former_smoker",
    "ever_smoked",
    "physically_active",
    "physically_inactive",
    "fair_poor_health",
    "college_education"
]


# ============================================================
# 12. CREATE FINAL FEATURE DATASET
# ============================================================

final_columns = (
    original_features +
    derived_features +
    ["stroke"]
)

result = df[final_columns].copy()


# ============================================================
# 13. REMOVE EXACT DUPLICATES
# ============================================================

# Some features may contain exactly the same information.
#
# We identify them here, but do NOT automatically remove them.
# Feature selection will handle redundancy later.

feature_only = result.drop(
    columns=["stroke"]
)

duplicate_features = []

columns = feature_only.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        if feature_only[columns[i]].equals(
            feature_only[columns[j]]
        ):
            duplicate_features.append(
                (columns[i], columns[j])
            )


print("\nExact Duplicate Feature Pairs:")

if duplicate_features:

    for pair in duplicate_features:
        print(pair)

else:

    print("None")


# ============================================================
# 14. DISPLAY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTED FEATURES")
print("=" * 70)

print("\nOriginal Features:")
for feature in original_features:
    print("  -", feature)

print("\nDerived Features:")
for feature in derived_features:
    print("  -", feature)

print("\nTotal Features:")
print(len(result.columns) - 1)


# ============================================================
# 15. CHECK OUTPUT
# ============================================================

print("\nOutput Shape:")
print(result.shape)

print("\nFirst 5 Rows:")
print(result.head())

print("\nMissing Values:")
print(result.isnull().sum())


# ============================================================
# 16. SAVE DATASET
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)