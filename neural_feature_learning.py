
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# NEURAL FEATURE LEARNING - BRFSS STROKE PROJECT
# Replaces:
#   1. feature_extraction.py
#   2. feature_selection.py
#
# The neural network learns a 15-dimensional bottleneck
# representation directly from the cleaned BRFSS variables.
# No handcrafted derived features, MI ranking, or RF ranking
# is used to create the final feature set.
# ============================================================

TRAIN_X_FILE = "X_train_brfss.csv"
TEST_X_FILE = "X_test_brfss.csv"
TRAIN_Y_FILE = "y_train_brfss.csv"
TEST_Y_FILE = "y_test_brfss.csv"

OUTPUT_TRAIN = "X_train_neural.csv"
OUTPUT_TEST = "X_test_neural.csv"
INPUT_ENCODER_FILE = "neural_input_encoder.joblib"
INPUT_SCALER_FILE = "neural_latent_scaler.joblib"
MODEL_FILE = "neural_feature_model.joblib"
FEATURE_NAMES_FILE = "neural_feature_names.csv"

LATENT_DIM = 15
RANDOM_STATE = 42


# ============================================================
# 1. LOAD PREPROCESSED TRAIN / TEST DATA
# ============================================================

print("=" * 70)
print("NEURAL FEATURE LEARNING - BRFSS STROKE PROJECT")
print("=" * 70)

X_train = pd.read_csv(TRAIN_X_FILE)
X_test = pd.read_csv(TEST_X_FILE)

y_train = pd.read_csv(TRAIN_Y_FILE).squeeze("columns")
y_test = pd.read_csv(TEST_Y_FILE).squeeze("columns")

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)
print("Training labels:")
print(y_train.value_counts())
print("\nTesting labels:")
print(y_test.value_counts())


# ============================================================
# 2. DEFINE FEATURE TYPES
# ============================================================
# These are the original cleaned BRFSS variables.
# We do not create the handcrafted features from feature_extraction.py.

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

missing_columns = [
    c for c in categorical_columns + numerical_columns
    if c not in X_train.columns
]

if missing_columns:
    raise ValueError(
        "Missing expected BRFSS columns: " + str(missing_columns)
    )


# ============================================================
# 3. ENCODE THE INPUT VARIABLES
# ============================================================
# Categorical BRFSS codes are one-hot encoded so the network
# does not incorrectly assume equal numeric distance between
# categories.
#
# Numerical columns were already scaled by preprocessing.py.

X_train_input = X_train.copy()
X_test_input = X_test.copy()

for col in categorical_columns:
    X_train_input[col] = X_train_input[col].astype(str)
    X_test_input[col] = X_test_input[col].astype(str)

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
        dtype=np.float32,
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", encoder, categorical_columns),
        ("numerical", "passthrough", numerical_columns),
    ],
    remainder="drop",
)

X_train_encoded = preprocessor.fit_transform(X_train_input)
X_test_encoded = preprocessor.transform(X_test_input)

X_train_encoded = np.asarray(X_train_encoded, dtype=np.float32)
X_test_encoded = np.asarray(X_test_encoded, dtype=np.float32)

print("\nEncoded input shape :", X_train_encoded.shape)
print("Neural latent size  :", LATENT_DIM)


# ============================================================
# 4. BUILD THE SUPERVISED BOTTLENECK NETWORK
# ============================================================
#
# Input
#   -> 64 neurons
#   -> 32 neurons
#   -> 15-neuron BOTTLENECK  <-- learned features
#   -> 16 neurons
#   -> Stroke probability
#
# The 15-neuron bottleneck is the replacement for the previous
# manually engineered + selected 15-feature representation.

nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, LATENT_DIM, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=1024,
    learning_rate_init=1e-3,
    max_iter=80,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=8,
    random_state=RANDOM_STATE,
    verbose=True,
)


# ============================================================
# 5. HANDLE CLASS IMBALANCE
# ============================================================
# Stroke is the minority class. Use balanced sample weights when
# supported by the installed scikit-learn version.

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

print("\nTraining neural feature learner...")

try:
    nn_model.fit(
        X_train_encoded,
        y_train,
        sample_weight=sample_weights,
    )
except TypeError:
    # Fallback for older scikit-learn releases without
    # MLPClassifier.fit(..., sample_weight=...).
    print(
        "\nInstalled scikit-learn does not support MLP sample_weight."
    )
    print("Falling back to a class-balanced training subset...")

    rng = np.random.RandomState(RANDOM_STATE)

    y_array = np.asarray(y_train)
    positive_idx = np.where(y_array == 1)[0]
    negative_idx = np.where(y_array == 0)[0]

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        raise ValueError("Training data must contain both classes.")

    # Use at most 4 negatives per positive for manageable training size.
    negative_count = min(
        len(negative_idx),
        len(positive_idx) * 4
    )

    sampled_negative_idx = rng.choice(
        negative_idx,
        size=negative_count,
        replace=False,
    )

    balanced_idx = np.concatenate([
        positive_idx,
        sampled_negative_idx,
    ])

    rng.shuffle(balanced_idx)

    nn_model.fit(
        X_train_encoded[balanced_idx],
        y_array[balanced_idx],
    )


print("\nNeural network training complete.")
print("Iterations:", nn_model.n_iter_)
print("Final loss:", nn_model.loss_)


# ============================================================
# 6. EXTRACT THE 15-DIMENSIONAL BOTTLENECK
# ============================================================
#
# sklearn does not expose a direct transform() for MLPClassifier,
# so we forward-propagate through the hidden layers and capture
# the activation of the 15-neuron bottleneck layer.

def extract_bottleneck(model, X, bottleneck_layer_index=2):
    activation = X

    # Exclude final output layer.
    for layer_index, (weights, bias) in enumerate(
        zip(model.coefs_[:-1], model.intercepts_[:-1])
    ):
        activation = activation @ weights + bias
        activation = np.maximum(activation, 0.0)  # ReLU

        if layer_index == bottleneck_layer_index:
            return activation.astype(np.float32)

    raise RuntimeError("Bottleneck layer was not found.")


latent_train = extract_bottleneck(
    nn_model,
    X_train_encoded,
)

latent_test = extract_bottleneck(
    nn_model,
    X_test_encoded,
)

if latent_train.shape[1] != LATENT_DIM:
    raise RuntimeError(
        f"Expected {LATENT_DIM} latent features, "
        f"got {latent_train.shape[1]}"
    )

print("\nLearned feature matrix:")
print("Train:", latent_train.shape)
print("Test :", latent_test.shape)


# ============================================================
# 7. SCALE THE LEARNED FEATURES
# ============================================================
# Scaling is only for the downstream classifiers. It does not
# perform feature selection.

latent_scaler = StandardScaler()

latent_train_scaled = latent_scaler.fit_transform(
    latent_train
)

latent_test_scaled = latent_scaler.transform(
    latent_test
)


# ============================================================
# 8. NAME THE LEARNED FEATURES
# ============================================================

latent_feature_names = [
    f"nn_feature_{i:02d}"
    for i in range(1, LATENT_DIM + 1)
]

X_train_neural = pd.DataFrame(
    latent_train_scaled,
    columns=latent_feature_names,
)

X_test_neural = pd.DataFrame(
    latent_test_scaled,
    columns=latent_feature_names,
)


# ============================================================
# 9. SAVE LEARNED FEATURES
# ============================================================

X_train_neural.to_csv(
    OUTPUT_TRAIN,
    index=False,
)

X_test_neural.to_csv(
    OUTPUT_TEST,
    index=False,
)

pd.DataFrame({
    "Feature": latent_feature_names
}).to_csv(
    FEATURE_NAMES_FILE,
    index=False,
)

joblib.dump(
    preprocessor,
    INPUT_ENCODER_FILE,
)

joblib.dump(
    latent_scaler,
    INPUT_SCALER_FILE,
)

joblib.dump(
    nn_model,
    MODEL_FILE,
)


# ============================================================
# 10. PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("LEARNED NEURAL FEATURES - FIRST 5 TRAINING ROWS")
print("=" * 70)

print(X_train_neural.head())

print("\n" + "=" * 70)
print("NEURAL FEATURE LEARNING COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print("  X_train_neural.csv")
print("  X_test_neural.csv")
print("  neural_input_encoder.joblib")
print("  neural_latent_scaler.joblib")
print("  neural_feature_model.joblib")
print("  neural_feature_names.csv")

print(
    "\nThe 15 nn_feature_* columns are learned latent features "
    "and replace the old handcrafted feature-extraction and "
    "MI/Random-Forest feature-selection stages."
)
