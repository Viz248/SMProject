import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE SCALING - BRFSS STROKE PROJECT
# ============================================================

TRAIN_X_FILE = "X_train_final.csv"
TEST_X_FILE = "X_test_final.csv"

OUTPUT_TRAIN = "X_train_scaled.csv"
OUTPUT_TEST = "X_test_scaled.csv"


# ============================================================
# 1. LOAD FINAL SELECTED FEATURES
# ============================================================

X_train = pd.read_csv(TRAIN_X_FILE)
X_test = pd.read_csv(TEST_X_FILE)

print("=" * 70)
print("FEATURE SCALING")
print("=" * 70)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nFeatures:")
print(list(X_train.columns))


# ============================================================
# 2. CREATE SCALER
# ============================================================

scaler = StandardScaler()


# ============================================================
# 3. FIT ONLY ON TRAINING DATA
# ============================================================
#
# IMPORTANT:
# The test data is NOT used to calculate mean/std.
#
# This prevents data leakage.

X_train_scaled = scaler.fit_transform(X_train)


# ============================================================
# 4. TRANSFORM TEST DATA
# ============================================================
#
# Use the scaler learned from training data.

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. CONVERT BACK TO DATAFRAMES
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)


# ============================================================
# 6. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SCALED TRAINING DATA - FIRST 5 ROWS")
print("=" * 70)

print(
    X_train_scaled.head()
)


print("\n" + "=" * 70)
print("SCALED TEST DATA - FIRST 5 ROWS")
print("=" * 70)

print(
    X_test_scaled.head()
)


# ============================================================
# 7. CHECK SCALING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE MEANS AFTER SCALING")
print("=" * 70)

print(
    X_train_scaled.mean()
)


print("\n" + "=" * 70)
print("TRAINING FEATURE STANDARD DEVIATIONS")
print("=" * 70)

print(
    X_train_scaled.std()
)


# ============================================================
# 8. SAVE
# ============================================================

X_train_scaled.to_csv(
    OUTPUT_TRAIN,
    index=False
)

X_test_scaled.to_csv(
    OUTPUT_TEST,
    index=False
)


print("\n" + "=" * 70)
print("FEATURE SCALING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("  X_train_scaled.csv")
print("  X_test_scaled.csv")
