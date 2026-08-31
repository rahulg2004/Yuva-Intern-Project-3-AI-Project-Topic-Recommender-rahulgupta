import os
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(CURRENT_FILE)
)

# Source directory
SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT PREPROCESSING FUNCTIONS
# ============================================================

from preprocessing import (
    load_data,
    clean_data,
    create_combined_features
)


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "project_topics.csv"
)


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    recommended_domains,
    target_domain,
    k
):
    """
    Calculate Precision@K.

    A recommendation is considered relevant
    when it belongs to the same domain as
    the target project.
    """

    recommendations = recommended_domains[:k]

    if len(recommendations) == 0:
        return 0.0

    relevant = sum(
        domain == target_domain
        for domain in recommendations
    )

    return relevant / len(recommendations)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(k=5):

    print("\n" + "=" * 60)
    print("AI PROJECT TOPIC RECOMMENDATION SYSTEM")
    print("MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Display paths
    # --------------------------------------------------------

    print("\nProject directory:")
    print(BASE_DIR)

    print("\nDataset location:")
    print(DATA_PATH)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        print("\nERROR: Dataset not found!")

        print("\nExpected location:")
        print(DATA_PATH)

        print(
            "\nMake sure the following file exists:"
        )

        print(
            "data/project_topics.csv"
        )

        return None

    # --------------------------------------------------------
    # STEP 1: Load dataset
    # --------------------------------------------------------

    print("\n1. Loading dataset...")

    df = load_data(
        DATA_PATH
    )

    print(
        f"   Loaded {len(df)} records."
    )

    # --------------------------------------------------------
    # STEP 2: Clean dataset
    # --------------------------------------------------------

    print("\n2. Cleaning dataset...")

    df = clean_data(df)

    print(
        f"   Cleaned dataset contains "
        f"{len(df)} records."
    )

    # --------------------------------------------------------
    # STEP 3: Create combined features
    # --------------------------------------------------------

    print("\n3. Creating combined features...")

    df = create_combined_features(
        df
    )

    print(
        "   Combined features created."
    )

    # --------------------------------------------------------
    # STEP 4: Create TF-IDF vectorizer
    # --------------------------------------------------------

    print("\n4. Creating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(

        stop_words="english",

        lowercase=True,

        ngram_range=(1, 2),

        max_features=5000,

        sublinear_tf=True
    )

    # --------------------------------------------------------
    # STEP 5: Transform data
    # --------------------------------------------------------

    print("\n5. Transforming project data...")

    matrix = vectorizer.fit_transform(
        df["combined_features"]
    )

    print(
        f"   Feature matrix shape: "
        f"{matrix.shape}"
    )

    # --------------------------------------------------------
    # STEP 6: Evaluate
    # --------------------------------------------------------

    print(
        f"\n6. Calculating Precision@{k}..."
    )

    scores = []

    for index in range(len(df)):

        # Use current project as query
        test_vector = matrix[index]

        # Calculate similarity with all projects
        similarities = cosine_similarity(
            test_vector,
            matrix
        ).flatten()

        # Don't recommend itself
        similarities[index] = -1

        # Sort from highest to lowest similarity
        ranked_indices = np.argsort(
            similarities
        )[::-1]

        # Get top K
        top_indices = ranked_indices[:k]

        # Get recommended project domains
        recommended_domains = [
            df.iloc[i]["domain"]
            for i in top_indices
        ]

        # Current project's domain
        target_domain = df.iloc[index]["domain"]

        # Calculate precision
        score = precision_at_k(
            recommended_domains,
            target_domain,
            k
        )

        scores.append(score)

    # --------------------------------------------------------
    # STEP 7: Calculate average score
    # --------------------------------------------------------

    average_precision = float(
        np.mean(scores)
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nPrecision@{k}: "
        f"{average_precision:.4f}"
    )

    print(
        f"Precision@{k} Percentage: "
        f"{average_precision * 100:.2f}%"
    )

    print(
        f"\nProjects evaluated: "
        f"{len(df)}"
    )

    print(
        f"Recommendations per query: "
        f"{k}"
    )

    print("\nEvaluation completed successfully.")

    print("=" * 60)

    return average_precision


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_model(k=5)
