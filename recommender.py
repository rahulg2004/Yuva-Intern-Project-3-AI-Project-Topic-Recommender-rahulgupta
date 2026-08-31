import os
import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = "models"

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


class ProjectRecommender:

    def __init__(self):

        self._check_model_files()

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        self.project_matrix = joblib.load(
            MATRIX_PATH
        )

        self.df = pd.read_pickle(
            DATAFRAME_PATH
        )

    def _check_model_files(self):

        files = [
            VECTORIZER_PATH,
            MATRIX_PATH,
            DATAFRAME_PATH
        ]

        missing_files = [
            file
            for file in files
            if not os.path.exists(file)
        ]

        if missing_files:

            raise FileNotFoundError(
                "Required model files are missing.\n"
                "Please run:\n\n"
                "python src/train_model.py\n\n"
                "Missing files:\n" +
                "\n".join(missing_files)
            )

    @staticmethod
    def clean_input(text):

        if text is None:
            return ""

        return str(text).strip()

    def create_user_profile(
        self,
        interests,
        skills,
        career_goal,
        domain,
        difficulty
    ):

        interests = self.clean_input(interests)
        skills = self.clean_input(skills)
        career_goal = self.clean_input(career_goal)
        domain = self.clean_input(domain)
        difficulty = self.clean_input(difficulty)

        profile = (
            f"{interests} "
            f"{skills} "
            f"{career_goal} "
            f"{domain} "
            f"{difficulty}"
        )

        return profile.strip()

    def recommend(
        self,
        interests,
        skills,
        career_goal,
        domain,
        difficulty,
        top_n=5
    ):

        user_profile = self.create_user_profile(
            interests,
            skills,
            career_goal,
            domain,
            difficulty
        )

        if not user_profile:

            return pd.DataFrame()

        # Convert user profile to TF-IDF vector
        user_vector = self.vectorizer.transform(
            [user_profile]
        )

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(
            user_vector,
            self.project_matrix
        ).flatten()

        results = self.df.copy()

        results["similarity_score"] = (
            similarity_scores
        )

        # Sort by similarity
        results = results.sort_values(
            by="similarity_score",
            ascending=False
        )

        # Remove unnecessary internal column
        output_columns = [
            "project_id",
            "title",
            "domain",
            "description",
            "skills",
            "career_goal",
            "difficulty",
            "similarity_score"
        ]

        results = results[
            output_columns
        ]

        # Limit number of recommendations
        results = results.head(top_n)

        # Convert similarity to percentage
        results["match_percentage"] = (
            results["similarity_score"] * 100
        )

        return results.reset_index(drop=True)

    def get_all_projects(self):

        columns = [
            "project_id",
            "title",
            "domain",
            "description",
            "skills",
            "career_goal",
            "difficulty"
        ]

        return self.df[columns].copy()

    def get_domains(self):

        return sorted(
            self.df["domain"]
            .dropna()
            .unique()
            .tolist()
        )

    def get_difficulties(self):

        return sorted(
            self.df["difficulty"]
            .dropna()
            .unique()
            .tolist()
        )