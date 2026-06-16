#!/usr/bin/env python3
"""
================================================================================
 ARTIFICIAL INTELLIGENCE P2 — Data Classification Using AI
================================================================================

 Author       : DecodeLabs — Machine Learning Engineer
 Project      : Transitioning from hard-coded rules (P1) to a supervised
                learning pipeline that discovers patterns in data.

 Dataset      : Fisher's Iris Dataset (1936)
                150 samples × 4 numeric features × 3 species classes
                Features: sepal length, sepal width, petal length, petal width
                Classes:  Setosa (0), Versicolor (1), Virginica (2)

 Pipeline:
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  STEP 1      │ ▸▸│  STEP 2      │ ▸▸│  STEP 3      │ ▸▸│  STEP 4      │
   │  Load Data   │   │  Scale       │   │  Split       │   │  Train &     │
   │  (Raw CSV)   │   │  (Standard-  │   │  (Shuffle &  │   │  Predict     │
   │              │   │   Scaler)    │   │   80/20)     │   │  (KNN)       │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘

 Design Principles:
   1. FROM SCRATCH  — StandardScaler and Train/Test split are implemented
                      manually (no sklearn shortcuts for these two) to
                      demonstrate understanding of the underlying math.
   2. TRANSPARENT   — Every formula is commented with its mathematical
                      derivation.
   3. REPRODUCIBLE  — Fixed random seed ensures identical results every run.
   4. EVALUATED     — Full classification report + accuracy metrics.

================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np                        # Linear algebra & array operations
from sklearn.datasets import load_iris    # Iris dataset loader
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm
from sklearn.metrics import (
    classification_report,                # Precision, recall, F1-score
    accuracy_score,                       # Overall accuracy
    confusion_matrix,                     # Confusion matrix
)
import warnings
warnings.filterwarnings('ignore')        # Suppress non-critical warnings


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42        # Lock randomness for full reproducibility
TRAIN_RATIO    = 0.80      # 80% of data → training, 20% → testing
K_NEIGHBORS    = 5         # Number of nearest neighbours for KNN classifier


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA HANDLING: Load and Inspect the Raw Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_and_inspect_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the Iris dataset and display its structure.

    The Iris dataset is a classic ML benchmark:
      - 150 samples (50 per class)
      - 4 features per sample (sepal length, sepal width, petal length, petal width)
      - 3 target classes (Setosa, Versicolor, Virginica)

    Returns:
        feature_matrix (np.ndarray): Shape (150, 4) — raw feature values.
        target_vector  (np.ndarray): Shape (150,)   — integer class labels (0, 1, 2).
        feature_names  (list[str]):  Human-readable feature names.
    """
    print("=" * 72)
    print("  STEP 1 — DATA HANDLING")
    print("=" * 72)

    # ── Load the dataset from sklearn's built-in repository ──────────
    iris = load_iris()

    # ── Extract the components ───────────────────────────────────────
    feature_matrix = iris.data          # (150, 4) numpy array of floats
    target_vector  = iris.target        # (150,) numpy array of ints: 0, 1, 2
    feature_names  = list(iris.feature_names)   # e.g., 'sepal length (cm)'
    class_names    = list(iris.target_names)     # e.g., 'setosa'

    # ── Print dataset summary ────────────────────────────────────────
    num_samples, num_features = feature_matrix.shape
    print(f"\n  📊 Dataset     : Fisher's Iris (1936)")
    print(f"  📏 Samples     : {num_samples}")
    print(f"  🔢 Features    : {num_features}  →  {feature_names}")
    print(f"  🏷️  Classes      : {len(class_names)}  →  {class_names}")
    print(f"\n  First 5 rows (raw features):")
    print(f"  {'Sepal L':>10} {'Sepal W':>10} {'Petal L':>10} {'Petal W':>10} {'Class':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for i in range(5):
        row = feature_matrix[i]
        label = class_names[target_vector[i]]
        print(f"  {row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>10.2f} {row[3]:>10.2f} {label:>8}")

    return feature_matrix, target_vector, feature_names, class_names


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA PREPROCESSING: StandardScaler (Z-Score Normalisation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_standard_scaler(
    X_train_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a StandardScaler on the TRAINING data and return the parameters.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND STANDARD SCALING (Z-Score Normalisation)       ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  PROBLEM:                                                         ║
    ║    Features live on different scales.                             ║
    ║    e.g., sepal length ≈ 5–7 cm, but petal width ≈ 0.1–2.5 cm     ║
    ║    Without scaling, larger-scaled features DOMINATE distance      ║
    ║    calculations (critical for KNN, which uses Euclidean distance).║
    ║                                                                   ║
    ║  SOLUTION: Transform each feature to have:                       ║
    ║    • Mean (μ)     = 0   (centred around zero)                    ║
    ║    • Std Dev (σ)  = 1   (unit variance)                          ║
    ║                                                                   ║
    ║  FORMULA:                                                         ║
    ║                    x_ij - μ_j                                     ║
    ║    x_scaled(i,j) = ───────────                                    ║
    ║                       σ_j                                         ║
    ║                                                                   ║
    ║  Where:                                                           ║
    ║    x_ij = raw value of sample i, feature j                       ║
    ║    μ_j  = mean of feature j (computed from TRAINING set only)    ║
    ║    σ_j  = std deviation of feature j  (from TRAINING set only)   ║
    ║                                                                   ║
    ║  WHY FIT ON TRAINING ONLY?                                        ║
    ║    In production, you won't have the test set's statistics.      ║
    ║    Using test data to compute μ/σ would be DATA LEAKAGE.         ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        X_train_raw: (n_train, 4) raw feature matrix for training data.

    Returns:
        feature_means  : (4,) mean of each feature computed from training data.
        feature_stds   : (4,) std deviation of each feature from training data.
        X_train_scaled : (n_train, 4) scaled training feature matrix.
    """
    print("\n" + "=" * 72)
    print("  STEP 2 — DATA PREPROCESSING: StandardScaler (Manual Implementation)")
    print("=" * 72)

    # ── Compute the MEAN of each feature (column) ────────────────────
    #    Formula: μ_j = (1/n) * Σ(x_ij) for i = 1..n
    #    axis=0 computes along rows → one mean per feature column
    feature_means = np.mean(X_train_raw, axis=0)

    # ── Compute the STANDARD DEVIATION of each feature ───────────────
    #    Formula: σ_j = sqrt( (1/n) * Σ(x_ij - μ_j)² )
    #    We use ddof=0 (population std) for consistency with sklearn.
    feature_stds = np.std(X_train_raw, axis=0, ddof=0)

    # ── Apply the Z-Score transformation ─────────────────────────────
    #    x_scaled(i,j) = (x_ij - μ_j) / σ_j
    #    Broadcasting: (150, 4) - (1, 4) → (150, 4), then ÷ (1, 4)
    X_train_scaled = (X_train_raw - feature_means) / feature_stds

    # ── Print the computed statistics ────────────────────────────────
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    print(f"\n  📐 Computed from TRAINING set only (to prevent data leakage):\n")
    print(f"  {'Feature':<15} {'Mean (μ)':>12} {'Std Dev (σ)':>12}")
    print(f"  {'─'*15} {'─'*12} {'─'*12}")
    for i, name in enumerate(feature_names):
        print(f"  {name:<15} {feature_means[i]:>12.4f} {feature_stds[i]:>12.4f}")

    # ── Verify: show that scaled data has μ ≈ 0, σ ≈ 1 ──────────────
    print(f"\n  ✅ Verification (scaled training data should have μ≈0, σ≈1):\n")
    scaled_means = np.mean(X_train_scaled, axis=0)
    scaled_stds  = np.std(X_train_scaled, axis=0, ddof=0)
    print(f"  {'Feature':<15} {'Scaled μ':>12} {'Scaled σ':>12}")
    print(f"  {'─'*15} {'─'*12} {'─'*12}")
    for i, name in enumerate(feature_names):
        print(f"  {name:<15} {scaled_means[i]:>12.4f} {scaled_stds[i]:>12.4f}")

    print(f"\n  📊 First 3 scaled samples:")
    print(f"  {'Sepal L':>10} {'Sepal W':>10} {'Petal L':>10} {'Petal W':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for i in range(3):
        row = X_train_scaled[i]
        print(f"  {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]:>10.4f}")

    return feature_means, feature_stds, X_train_scaled


def apply_scaler(
    X_raw: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """
    Apply a previously fitted StandardScaler to new data (e.g., the test set).

    Uses the SAME μ and σ computed from the training set — this is critical
    to simulate a real production scenario.

    Args:
        X_raw  : (n, 4) raw feature matrix to transform.
        means  : (4,) feature means from the training set.
        stds   : (4,) feature stds  from the training set.

    Returns:
        (n, 4) scaled feature matrix.
    """
    return (X_raw - means) / stds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STRUCTURAL INTEGRITY: Shuffle & Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════

def shuffle_and_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle the dataset and split it into training and test sets.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND THE TRAIN/TEST SPLIT                            ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  WHY SHUFFLE?                                                     ║
    ║    The Iris dataset is ordered: samples 0–49 = Setosa,            ║
    ║    50–99 = Versicolor, 100–149 = Virginica.                       ║
    ║    Without shuffling, a naive split would put ALL of one class    ║
    ║    in training and NONE in testing → model can never learn it.   ║
    ║                                                                   ║
    ║  WHY SPLIT?                                                       ║
    ║    We need UNSEEN data to evaluate true generalisation.           ║
    ║    Training on data and testing on the same data would give      ║
    ║    inflated accuracy (overfitting illusion).                      ║
    ║                                                                   ║
    ║  SPLIT MATH:                                                      ║
    ║    n_total     = 150                                              ║
    ║    n_train     = floor(n_total × train_ratio) = floor(120) = 120 ║
    ║    n_test      = n_total - n_train = 30                          ║
    ║                                                                   ║
    ║  STRATIFICATION:                                                  ║
    ║    Since the dataset is balanced (50 per class) and we shuffle   ║
    ║    globally, each split will approximately preserve class         ║
    ║    proportions by chance. For production systems, explicit        ║
    ║    stratified splitting is preferred.                             ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        X           : (n, 4) feature matrix (already scaled).
        y           : (n,)   target vector.
        train_ratio : Fraction of data to use for training (default 80%).
        seed        : Random seed for reproducibility.

    Returns:
        X_train, y_train, X_test, y_test
    """
    print("\n" + "=" * 72)
    print("  STEP 3 — STRUCTURAL INTEGRITY: Shuffle & Train/Test Split")
    print("=" * 72)

    n_total = X.shape[0]

    # ── Step 3a: SHUFFLE the data to remove ordering bias ────────────
    #    Generate a random permutation of indices [0, 1, 2, ..., 149]
    #    and reorder both X and y by the same permutation.
    rng = np.random.default_rng(seed)       # Reproducible RNG
    shuffled_indices = rng.permutation(n_total)

    X_shuffled = X[shuffled_indices]        # Reorder rows of features
    y_shuffled = y[shuffled_indices]        # Reorder rows of targets

    # ── Step 3b: SPLIT at the boundary ───────────────────────────────
    #    n_train = floor(150 × 0.80) = 120
    #    First 120 shuffled samples → training
    #    Last  30 shuffled samples → testing
    n_train = int(np.floor(n_total * train_ratio))
    n_test  = n_total - n_train

    X_train = X_shuffled[:n_train]          # Rows 0..119
    y_train = y_shuffled[:n_train]

    X_test  = X_shuffled[n_train:]          # Rows 120..149
    y_test  = y_shuffled[n_train:]

    # ── Print the split summary ──────────────────────────────────────
    print(f"\n  🎲 Shuffled dataset with seed = {seed}")
    print(f"  ✂️  Split ratio : {train_ratio:.0%} train / {1-train_ratio:.0%} test")
    print(f"  📦 Training set : {X_train.shape[0]} samples  (shape {X_train.shape})")
    print(f"  📦 Test set     : {X_test.shape[0]} samples   (shape {X_test.shape})")

    # ── Verify class distribution in each split ──────────────────────
    class_names = ["Setosa", "Versicolor", "Virginica"]
    print(f"\n  📈 Class distribution:\n")
    print(f"  {'Class':<15} {'Train Count':>12} {'Test Count':>12} {'Total':>8}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*8}")
    for cls_idx, cls_name in enumerate(class_names):
        train_count = np.sum(y_train == cls_idx)
        test_count  = np.sum(y_test == cls_idx)
        total       = train_count + test_count
        print(f"  {cls_name:<15} {train_count:>12} {test_count:>12} {total:>8}")

    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ALGORITHM IMPLEMENTATION: K-Nearest Neighbours (KNN)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
    class_names: list[str] = None,
) -> dict:
    """
    Train a K-Nearest Neighbours classifier and evaluate on the test set.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND K-NEAREST NEIGHBOURS (KNN)                      ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  OVERVIEW:                                                        ║
    ║    KNN is a "lazy learner" — it stores the entire training set   ║
    ║    and, for each new test sample, finds the K closest training   ║
    ║    samples and takes a majority vote on their labels.             ║
    ║                                                                   ║
    ║  DISTANCE METRIC — Euclidean Distance:                           ║
    ║                                                                   ║
    ║    d(a, b) = sqrt( Σ(a_j - b_j)² )   for j = 1..features       ║
    ║                                                                   ║
    ║    For two 4D points (Iris features):                            ║
    ║    d = sqrt((a₁-b₁)² + (a₂-b₂)² + (a₃-b₃)² + (a₄-b₄)²)       ║
    ║                                                                   ║
    ║  WHY SCALING MATTERS FOR KNN:                                    ║
    ║    Without scaling, features with larger magnitudes would        ║
    ║    dominate the distance. Scaling to μ=0, σ=1 ensures all       ║
    ║    features contribute EQUALLY to the distance calculation.      ║
    ║                                                                   ║
    ║  PREDICTION:                                                      ║
    ║    For test sample x_new:                                         ║
    ║      1. Compute d(x_new, x_i) for all training samples           ║
    ║      2. Sort by distance, take the K smallest                    ║
    ║      3. Count labels among those K neighbours                    ║
    ║      4. Assign the MAJORITY label                                ║
    ║                                                                   ║
    ║  WHY K=5?                                                         ║
    ║    Odd K prevents ties in binary classification.                  ║
    ║    K=5 is a common heuristic: small enough to capture local      ║
    ║    structure, large enough to smooth noise.                      ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        X_train     : (n_train, 4) scaled training features.
        y_train     : (n_train,)   training labels.
        X_test      : (n_test, 4)  scaled test features.
        y_test      : (n_test,)    true test labels.
        k           : Number of neighbours (default 5).
        class_names : List of human-readable class names.

    Returns:
        results (dict): Contains predictions, accuracy, and report strings.
    """
    print("\n" + "=" * 72)
    print("  STEP 4 — ALGORITHM: K-Nearest Neighbours (KNN)")
    print("=" * 72)

    if class_names is None:
        class_names = ["Setosa", "Versicolor", "Virginica"]

    # ── Step 4a: INITIALISE the KNN classifier ───────────────────────
    knn = KNeighborsClassifier(
        n_neighbors=k,           # K = 5 nearest neighbours
        metric='euclidean',      # L2 distance: sqrt(Σ(a-b)²)
        weights='uniform',       # All K neighbours vote equally
        algorithm='auto',        # Let sklearn pick the best search algo
    )

    # ── Step 4b: TRAIN (fit) the model ───────────────────────────────
    #    KNN's "training" simply stores X_train and y_train.
    #    No gradient descent, no weight updates — it's a lazy learner.
    print(f"\n  🏋️ Training KNN classifier (K={k})...")
    knn.fit(X_train, y_train)
    print(f"  ✅ Model fitted. Training samples stored: {X_train.shape[0]}")

    # ── Step 4c: PREDICT labels for the test set ─────────────────────
    #    For each test sample, the model:
    #      1. Computes Euclidean distance to ALL 120 training samples
    #      2. Sorts distances, picks the 5 smallest
    #      3. Takes a majority vote among those 5 neighbours' labels
    y_pred = knn.predict(X_test)

    # ── Step 4d: EVALUATE performance ────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print(f"\n  🎯 PREDICTIONS vs ACTUAL (test set):")
    print(f"  {'Index':>6} {'Predicted':<14} {'Actual':<14} {'Match':>6}")
    print(f"  {'─'*6} {'─'*14} {'─'*14} {'─'*6}")
    for i in range(len(y_test)):
        pred_label = class_names[y_pred[i]]
        true_label = class_names[y_test[i]]
        match = "  ✅" if y_pred[i] == y_test[i] else "  ❌"
        print(f"  {i:>6} {pred_label:<14} {true_label:<14} {match}")

    print(f"\n  {'━'*50}")
    print(f"  📊 ACCURACY : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  {'━'*50}")

    print(f"\n  📋 CLASSIFICATION REPORT:")
    print(f"  ┌{'─'*58}┐")
    for line in report.split("\n"):
        print(f"  │ {line:<57}│")
    print(f"  └{'─'*58}┘")

    print(f"\n  🔢 CONFUSION MATRIX:")
    print(f"  (Rows = Actual, Columns = Predicted)\n")
    header = f"  {'':>15}" + "".join(f"{c:>12}" for c in class_names)
    print(header)
    print(f"  {'─'*51}")
    for i, cls in enumerate(class_names):
        row_str = f"  {cls:>15}" + "".join(f"{conf_matrix[i][j]:>12}" for j in range(len(class_names)))
        print(row_str)

    # ── Return structured results ────────────────────────────────────
    return {
        "model": knn,
        "predictions": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — SINGLE-SAMPLE PREDICTION (Inference Demo)
# ══════════════════════════════════════════════════════════════════════════════

def predict_single_sample(
    model: KNeighborsClassifier,
    means: np.ndarray,
    stds: np.ndarray,
    raw_features: list[float],
    class_names: list[str],
) -> None:
    """
    Demonstrate inference on a single, unseen sample.

    This shows the COMPLETE pipeline for one data point:
      1. Accept raw measurements (like a user would type)
      2. Apply the SAME scaling transform (using training μ and σ)
      3. Pass through the trained model
      4. Print the predicted class

    Args:
        model       : The fitted KNN classifier.
        means       : Feature means from training set.
        stds        : Feature stds from training set.
        raw_features: [sepal_len, sepal_wid, petal_len, petal_wid] in cm.
        class_names : Human-readable class names.
    """
    print("\n" + "=" * 72)
    print("  BONUS — SINGLE-SAMPLE INFERENCE")
    print("=" * 72)

    raw = np.array(raw_features).reshape(1, -1)   # (1, 4)

    # ── Scale using the SAME μ and σ from training ───────────────────
    scaled = apply_scaler(raw, means, stds)

    # ── Predict ──────────────────────────────────────────────────────
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]

    print(f"\n  🌸 Raw input       : {raw_features}")
    print(f"  📐 Scaled input    : {scaled[0].round(4)}")
    print(f"  🏷️  Predicted class : {class_names[prediction]}  (label {prediction})")
    print(f"\n  📊 Neighbour vote probabilities:")
    for i, cls in enumerate(class_names):
        bar = "█" * int(probabilities[i] * 30)
        print(f"     {cls:<14} {probabilities[i]:.2%}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — ORCHESTRATE THE FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate the complete ML pipeline end-to-end.

    Flow:
      Step 1: Load raw data
      Step 2: Fit scaler on training data → transform both sets
      Step 3: Shuffle & split into train/test
      Step 4: Train KNN, predict on test, evaluate
      Bonus : Predict on a brand-new sample
    """

    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " ARTIFICIAL INTELLIGENCE P2 — Data Classification Using AI".center(70) + "║")
    print("║" + " DecodeLabs • Supervised Learning Pipeline (KNN)".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    # ── STEP 1: Load Data ────────────────────────────────────────────
    X_raw, y, feature_names, class_names = load_and_inspect_data()

    # ── STEP 3 FIRST: Split RAW data BEFORE scaling ──────────────────
    #    This ordering is critical: we must split first, then fit the
    #    scaler ONLY on training data to prevent data leakage.
    X_train_raw, y_train, X_test_raw, y_test = shuffle_and_split(
        X_raw, y, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED,
    )

    # ── STEP 2: Fit Scaler on Training Data Only ─────────────────────
    means, stds, X_train_scaled = compute_standard_scaler(X_train_raw)

    # ── Apply the SAME scaler to the test set ────────────────────────
    X_test_scaled = apply_scaler(X_test_raw, means, stds)
    print(f"\n  ✅ Applied same μ/σ to test set (no data leakage).")

    # ── STEP 4: Train & Evaluate ─────────────────────────────────────
    results = train_and_predict(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        k=K_NEIGHBORS,
        class_names=class_names,
    )

    # ── BONUS: Single-Sample Inference ───────────────────────────────
    #    Simulate a user typing in new measurements.
    #    Example: a flower with sepal_len=5.1, sepal_wid=3.5,
    #             petal_len=1.4, petal_wid=0.2 → should be Setosa.
    predict_single_sample(
        model=results["model"],
        means=means,
        stds=stds,
        raw_features=[5.1, 3.5, 1.4, 0.2],
        class_names=class_names,
    )

    # ── Second bonus sample ──────────────────────────────────────────
    predict_single_sample(
        model=results["model"],
        means=means,
        stds=stds,
        raw_features=[6.7, 3.1, 4.7, 1.5],  # Should be Versicolor
        class_names=class_names,
    )

    # ── Pipeline Complete ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 72)
    print(f"""
  Summary of what happened:
    1. Loaded 150 Iris samples with 4 numeric features each.
    2. Split into 120 training / 30 test samples (shuffled, seed=42).
    3. Fitted StandardScaler on training data: μ_j, σ_j per feature.
    4. Transformed both sets: x_scaled = (x - μ) / σ
    5. Trained KNN (K=5, Euclidean distance) on scaled training data.
    6. Predicted on 30 test samples.
    7. Achieved {results['accuracy']:.1%} accuracy.
    8. Demonstrated single-sample inference on new data points.
""")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
