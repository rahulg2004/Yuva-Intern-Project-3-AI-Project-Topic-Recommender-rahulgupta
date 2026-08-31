import streamlit as st

from src.recommender import ProjectRecommender


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Project Topic Recommender",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_recommender():

    return ProjectRecommender()


try:

    recommender = load_recommender()

except FileNotFoundError as error:

    st.error(
        "The recommendation model has not been trained yet."
    )

    st.code(
        "python src/train_model.py",
        language="powershell"
    )

    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🤖 AI Project Topic Recommender")

st.write(
    "Get personalized project recommendations based on "
    "your interests, technical skills and career goals."
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses a content-based "
        "recommendation system."
    )

    st.write(
        "TF-IDF is used for feature extraction and "
        "Cosine Similarity is used to calculate "
        "project relevance."
    )

    st.divider()

    st.subheader("Technology")

    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("TF-IDF")
    st.write("Cosine Similarity")
    st.write("Streamlit")


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("Tell us about yourself")

col1, col2 = st.columns(2)


with col1:

    interests = st.text_area(
        "Your Interests",
        placeholder=(
            "Example: Artificial Intelligence, "
            "Computer Vision, Generative AI"
        ),
        height=120
    )

    skills = st.text_area(
        "Your Technical Skills",
        placeholder=(
            "Example: Python, OpenCV, "
            "Machine Learning, Pandas"
        ),
        height=120
    )

    career_goal = st.text_input(
        "Career Goal",
        placeholder="Example: AI Engineer"
    )


with col2:

    domains = recommender.get_domains()

    domain = st.selectbox(
        "Preferred Domain",
        domains
    )

    difficulties = recommender.get_difficulties()

    difficulty = st.selectbox(
        "Preferred Difficulty",
        difficulties
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )


st.divider()


# ---------------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------------

recommend_button = st.button(
    "🚀 Recommend Projects",
    use_container_width=True
)


if recommend_button:

    # Validate input

    if not interests.strip():

        st.warning(
            "Please enter your interests."
        )

    elif not skills.strip():

        st.warning(
            "Please enter your technical skills."
        )

    elif not career_goal.strip():

        st.warning(
            "Please enter your career goal."
        )

    else:

        recommendations = recommender.recommend(
            interests=interests,
            skills=skills,
            career_goal=career_goal,
            domain=domain,
            difficulty=difficulty,
            top_n=top_n
        )

        if recommendations.empty:

            st.error(
                "No recommendations could be generated."
            )

        else:

            st.success(
                f"Found {len(recommendations)} "
                "personalized recommendations."
            )

            st.subheader(
                "🎯 Recommended Projects"
            )

            # Display each recommendation

            for index, row in recommendations.iterrows():

                score = row["match_percentage"]

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### {index + 1}. "
                        f"{row['title']}"
                    )

                    info_col1, info_col2, info_col3 = (
                        st.columns(3)
                    )

                    with info_col1:

                        st.write(
                            f"**Domain:** "
                            f"{row['domain']}"
                        )

                    with info_col2:

                        st.write(
                            f"**Difficulty:** "
                            f"{row['difficulty']}"
                        )

                    with info_col3:

                        st.write(
                            f"**Career:** "
                            f"{row['career_goal']}"
                        )

                    st.write(
                        f"**Required Skills:** "
                        f"{row['skills']}"
                    )

                    st.write(
                        row["description"]
                    )

                    st.write(
                        f"**Match Score: "
                        f"{score:.2f}%**"
                    )

                    # Progress bar
                    progress_value = max(
                        0.0,
                        min(
                            float(row["similarity_score"]),
                            1.0
                        )
                    )

                    st.progress(
                        progress_value
                    )


# ---------------------------------------------------------
# PROJECT DATABASE
# ---------------------------------------------------------

st.divider()

with st.expander(
    "📚 View Available Project Dataset"
):

    all_projects = (
        recommender.get_all_projects()
    )

    st.dataframe(
        all_projects,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI Project Topic Recommendation System | "
    "TF-IDF + Cosine Similarity"
)