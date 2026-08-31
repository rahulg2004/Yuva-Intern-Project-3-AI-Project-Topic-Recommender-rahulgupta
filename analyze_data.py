import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

BASE_DIR = os.path.dirname(
    os.path.dirname(CURRENT_FILE)
)


# ============================================================
# DIRECTORIES
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "project_topics.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AI PROJECT TOPIC RECOMMENDER")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

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
            "\nExpected location:"
        )

        print(DATA_PATH)

        return

    # --------------------------------------------------------
    # Create analysis directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\n1. Loading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"   Loaded {len(df)} records."
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n2. Dataset Information")

    print(
        f"   Number of records: "
        f"{len(df)}"
    )

    print(
        f"   Number of columns: "
        f"{len(df.columns)}"
    )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\n3. Columns")

    for column in df.columns:

        print(
            f"   - {column}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n4. Missing Values")

    missing_values = df.isnull().sum()

    print(
        missing_values
    )

    # --------------------------------------------------------
    # Duplicate records
    # --------------------------------------------------------

    print("\n5. Duplicate Records")

    duplicate_count = df.duplicated().sum()

    print(
        f"   {duplicate_count}"
    )

    # --------------------------------------------------------
    # Domain distribution
    # --------------------------------------------------------

    print("\n6. Domain Distribution")

    domain_counts = (
        df["domain"]
        .value_counts()
    )

    print(
        domain_counts
    )

    # --------------------------------------------------------
    # Difficulty distribution
    # --------------------------------------------------------

    print("\n7. Difficulty Distribution")

    difficulty_counts = (
        df["difficulty"]
        .value_counts()
    )

    print(
        difficulty_counts
    )

    # --------------------------------------------------------
    # Career distribution
    # --------------------------------------------------------

    print("\n8. Career Goal Distribution")

    career_counts = (
        df["career_goal"]
        .value_counts()
    )

    print(
        career_counts
    )

    # ========================================================
    # CREATE DOMAIN GRAPH
    # ========================================================

    print(
        "\n9. Creating domain distribution chart..."
    )

    plt.figure(
        figsize=(10, 6)
    )

    domain_counts.plot(
        kind="bar"
    )

    plt.title(
        "Project Distribution by Domain"
    )

    plt.xlabel(
        "Domain"
    )

    plt.ylabel(
        "Number of Projects"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    domain_chart = os.path.join(
        OUTPUT_DIR,
        "domain_distribution.png"
    )

    plt.savefig(
        domain_chart,
        dpi=300
    )

    plt.close()

    print(
        f"   Saved: {domain_chart}"
    )

    # ========================================================
    # CREATE DIFFICULTY GRAPH
    # ========================================================

    print(
        "\n10. Creating difficulty distribution chart..."
    )

    plt.figure(
        figsize=(8, 5)
    )

    difficulty_counts.plot(
        kind="bar"
    )

    plt.title(
        "Project Distribution by Difficulty"
    )

    plt.xlabel(
        "Difficulty"
    )

    plt.ylabel(
        "Number of Projects"
    )

    plt.tight_layout()

    difficulty_chart = os.path.join(
        OUTPUT_DIR,
        "difficulty_distribution.png"
    )

    plt.savefig(
        difficulty_chart,
        dpi=300
    )

    plt.close()

    print(
        f"   Saved: {difficulty_chart}"
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 60)
    print("DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nAnalysis folder:")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()