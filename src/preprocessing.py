import os
import pandas as pd


# Get the main project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "project_topics.csv"
)


TEXT_COLUMNS = [
    "title",
    "domain",
    "description",
    "skills",
    "career_goal",
    "difficulty"
]


def load_data(path=DATA_PATH):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at:\n{path}"
        )

    return pd.read_csv(path)


def clean_data(df):

    df = df.copy()

    df = df.drop_duplicates()

    required_columns = [
        "title",
        "domain",
        "description",
        "skills",
        "career_goal",
        "difficulty"
    ]

    df = df.dropna(
        subset=required_columns
    )

    for column in TEXT_COLUMNS:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(
                r"\s+",
                " ",
                regex=True
            )
        )

    return df


def create_combined_features(df):

    df = df.copy()

    df["combined_features"] = (
        df["title"] + " " +
        df["domain"] + " " +
        df["description"] + " " +
        df["skills"] + " " +
        df["career_goal"] + " " +
        df["difficulty"]
    )

    return df


def analyze_data(df):

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(
        f"\nNumber of records: {len(df)}"
    )

    print(
        f"Number of columns: {len(df.columns)}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f" - {column}")

    print("\nMissing Values:")

    print(df.isnull().sum())

    print("\nDuplicate Records:")

    print(df.duplicated().sum())

    print("\nProject Distribution by Domain:")

    print(
        df["domain"].value_counts()
    )

    print("\nProject Distribution by Difficulty:")

    print(
        df["difficulty"].value_counts()
    )

    print("\nCareer Goal Distribution:")

    print(
        df["career_goal"].value_counts()
    )

    print("\nDataset Preview:")

    print(df.head())


if __name__ == "__main__":

    print(
        f"Looking for dataset at:\n{DATA_PATH}"
    )

    df = load_data()

    print(
        "\nOriginal dataset loaded successfully."
    )

    df = clean_data(df)

    print(
        "Dataset cleaned successfully."
    )

    df = create_combined_features(df)

    analyze_data(df)
