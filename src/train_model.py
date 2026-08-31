import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

# Absolute path of this file:
# .../AI-Project-Topic-Recommender/src/train_model.py

CURRENT_FILE = os.path.abspath(__file__)

# Go one level up from "src" to project root
BASE_DIR = os.path.dirname(
    os.path.dirname(CURRENT_FILE)
)

# Add src directory to Python path
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
# PROJECT DIRECTORIES
# ============================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    DATA_DIR,
    "project_topics.csv"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "project_matrix.pkl"
)

DATAFRAME_PATH = os.path.join(
    MODEL_DIR,
    "processed_projects.pkl"
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model():

    print("\n" + "=" * 60)
    print("AI PROJECT TOPIC RECOMMENDATION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Display project location
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

        print(
            "\nExpected dataset location:"
        )

        print(DATA_PATH)

        print(
            "\nPlease make sure your project has:"
        )

        print(
            "data/project_topics.csv"
        )

        return

    # --------------------------------------------------------
    # Create models directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

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

    original_count = len(df)

    df = clean_data(df)

    cleaned_count = len(df)

    print(
        f"   Original records: {original_count}"
    )

    print(
        f"   Cleaned records: {cleaned_count}"
    )

    print(
        f"   Removed records: "
        f"{original_count - cleaned_count}"
    )

    # --------------------------------------------------------
    # STEP 3: Create combined features
    # --------------------------------------------------------

    print("\n3. Creating combined features...")

    df = create_combined_features(
        df
    )

    print(
        "   Combined text features created."
    )

    # --------------------------------------------------------
    # STEP 4: Create TF-IDF Vectorizer
    # --------------------------------------------------------

    print("\n4. Creating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(

        # Remove common English words
        stop_words="english",

        # Convert text to lowercase
        lowercase=True,

        # Use single words and two-word combinations
        ngram_range=(1, 2),

        # Maximum number of features
        max_features=5000,

        # Improve TF-IDF weighting
        sublinear_tf=True
    )

    print(
        "   TF-IDF vectorizer created."
    )

    # --------------------------------------------------------
    # STEP 5: Transform dataset
    # --------------------------------------------------------

    print("\n5. Transforming project data...")

    project_matrix = vectorizer.fit_transform(
        df["combined_features"]
    )

    print(
        "   Feature matrix created."
    )

    print(
        f"   Matrix shape: "
        f"{project_matrix.shape}"
    )

    # --------------------------------------------------------
    # STEP 6: Save vectorizer
    # --------------------------------------------------------

    print("\n6. Saving TF-IDF vectorizer...")

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print(
        f"   Saved to: {VECTORIZER_PATH}"
    )

    # --------------------------------------------------------
    # STEP 7: Save project matrix
    # --------------------------------------------------------

    print("\n7. Saving project matrix...")

    joblib.dump(
        project_matrix,
        MATRIX_PATH
    )

    print(
        f"   Saved to: {MATRIX_PATH}"
    )

    # --------------------------------------------------------
    # STEP 8: Save processed dataset
    # --------------------------------------------------------

    print("\n8. Saving processed dataset...")

    df.to_pickle(
        DATAFRAME_PATH
    )

    print(
        f"   Saved to: {DATAFRAME_PATH}"
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated model files:")

    print(
        "✓ tfidf_vectorizer.pkl"
    )

    print(
        "✓ project_matrix.pkl"
    )

    print(
        "✓ processed_projects.pkl"
    )

    print("\nModel directory:")

    print(
        MODEL_DIR
    )

    print("\nYou can now run the application using:")

    print(
        "streamlit run app.py"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()
