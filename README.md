# 🤖 AI Project Topic Recommender

An AI-powered project recommendation system that recommends suitable project topics to students based on their **interests, technical skills, career goals, preferred domain, and difficulty level**.

The system uses **Natural Language Processing (NLP)** techniques, **TF-IDF Vectorization**, and **Cosine Similarity** to compare a student's profile with available project topics and generate personalized recommendations.

---

## 📌 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [🎯 Objective](#-objective)
- [✨ Key Features](#-key-features)
- [🧠 How the System Works](#-how-the-system-works)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 Data Flow](#-data-flow)
- [🛠️ Technologies Used](#️-technologies-used)
- [📂 Project Structure](#-project-structure)
- [📊 Dataset](#-dataset)
- [🧹 Data Preprocessing](#-data-preprocessing)
- [🔢 TF-IDF Vectorization](#-tf-idf-vectorization)
- [📐 Cosine Similarity](#-cosine-similarity)
- [⚙️ Installation](#️-installation)
- [▶️ Running the Project](#️-running-the-project)
- [📈 Data Analysis](#-data-analysis)
- [🤖 Model Training](#-model-training)
- [📊 Model Evaluation](#-model-evaluation)
- [🖥️ Streamlit Application](#️-streamlit-application)
- [🧪 Example Input](#-example-input)
- [🎯 Example Output](#-example-output)
- [🔐 Security Considerations](#-security-considerations)
- [📈 Scalability](#-scalability)
- [🔧 Maintainability](#-maintainability)
- [🚀 Future Enhancements](#-future-enhancements)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 Learning Outcomes](#-learning-outcomes)
- [👨‍💻 Author](#-author)
- [📄 License](#-license)

---

# 🌟 Project Overview

Choosing the right project topic can be difficult for students because there are thousands of possible project ideas across different technologies and domains.

The **AI Project Topic Recommender** solves this problem by analyzing the student's:

- 💡 Interests
- 🧑‍💻 Technical skills
- 🎯 Career goals
- 🏢 Preferred domain
- 📊 Preferred difficulty level

The system compares this information with project descriptions stored in a dataset and recommends the most relevant project topics.

The recommendation engine follows a **content-based recommendation approach**.

Instead of depending on ratings or previous users, the system focuses on the similarity between the student's profile and the information associated with each project.

---

# 🎯 Objective

The primary objectives of this project are:

1. 🤖 Build an AI-based project recommendation system.
2. 📊 Analyze and preprocess project-related data.
3. 🧹 Clean and prepare textual data for machine learning.
4. 📝 Convert project descriptions into numerical representations.
5. 🔢 Use TF-IDF to extract important textual features.
6. 📐 Use Cosine Similarity to calculate project relevance.
7. 🎯 Generate personalized project recommendations.
8. 📈 Evaluate recommendation performance.
9. 🖥️ Build an interactive Streamlit web application.
10. 🚀 Create a scalable foundation for future recommendation systems.

---

# ✨ Key Features

## 👤 Personalized Recommendations

The system generates recommendations according to individual student profiles.

It considers:

- Interests
- Skills
- Career goals
- Domain
- Difficulty

---

## 🧠 NLP-Based Recommendation

The system uses Natural Language Processing techniques to understand textual information.

Project information is converted into numerical vectors using:

**TF-IDF Vectorization**

---

## 📐 Similarity-Based Ranking

Cosine Similarity measures how closely the student's profile matches each project.

Projects with higher similarity scores are ranked higher.

---

## 📊 Data Analysis

The project includes an exploratory data analysis module that examines:

- Dataset size
- Dataset columns
- Missing values
- Duplicate records
- Domain distribution
- Difficulty distribution
- Career goal distribution

---

## 🧹 Data Preprocessing

The preprocessing pipeline performs:

- Duplicate removal
- Missing-value handling
- Text cleaning
- Whitespace normalization
- Feature combination

---

## 💾 Model Persistence

The trained components are saved using `joblib`.

Saved files include:

- `tfidf_vectorizer.pkl`
- `project_matrix.pkl`
- `processed_projects.pkl`

This avoids retraining the system every time the application starts.

---

## 🖥️ Interactive Web Interface

A Streamlit-based interface allows users to enter their information and receive recommendations without interacting with Python code directly.

---

# 🧠 How the System Works

The complete recommendation process can be summarized as:

```text
👤 Student Profile
       │
       ▼
📝 User Input
       │
       ▼
🧹 Data Cleaning
       │
       ▼
🔤 Text Combination
       │
       ▼
🔢 TF-IDF Vectorization
       │
       ▼
📐 Cosine Similarity
       │
       ▼
📊 Similarity Ranking
       │
       ▼
🎯 Top-N Recommendations
       │
       ▼
🖥️ Streamlit Interface
````

---

# 🏗️ System Architecture

```text
                    ┌───────────────────────┐
                    │       Student         │
                    │        Profile        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     User Interface    │
                    │      Streamlit        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Input Processing    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   TF-IDF Vectorizer   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Project Feature      │
                    │       Matrix          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Cosine Similarity    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Similarity Ranking    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Top-N Recommendations │
                    └───────────────────────┘
```

---

# 🔄 Data Flow

The system follows the following data flow:

### Step 1️⃣: Dataset Collection

Project information is stored in:

```text
data/project_topics.csv
```

### Step 2️⃣: Data Cleaning

The dataset is cleaned using the preprocessing module.

### Step 3️⃣: Feature Creation

Important text fields are combined into a single feature:

```text
Title
+
Domain
+
Description
+
Skills
+
Career Goal
+
Difficulty
```

### Step 4️⃣: TF-IDF Conversion

The combined text is converted into numerical vectors.

### Step 5️⃣: User Profile Creation

The student's inputs are combined into a profile.

### Step 6️⃣: Similarity Calculation

The user's profile vector is compared against all project vectors.

### Step 7️⃣: Ranking

Projects are sorted according to their similarity scores.

### Step 8️⃣: Recommendation

The top-N projects are displayed to the student.

---

# 🛠️ Technologies Used

| Technology           | Purpose                   |
| -------------------- | ------------------------- |
| 🐍 Python            | Core programming language |
| 🐼 Pandas            | Data processing           |
| 🔢 NumPy             | Numerical operations      |
| 🧠 Scikit-learn      | Machine learning and NLP  |
| 📝 TF-IDF            | Text feature extraction   |
| 📐 Cosine Similarity | Similarity calculation    |
| 💾 Joblib            | Model persistence         |
| 📊 Matplotlib        | Data visualization        |
| 🖥️ Streamlit        | Web application           |
| 📄 CSV               | Dataset storage           |

---

# 📂 Project Structure

```text
AI-Project-Topic-Recommender/
│
├── 📁 data/
│   └── project_topics.csv
│
├── 📁 models/
│   ├── tfidf_vectorizer.pkl
│   ├── project_matrix.pkl
│   └── processed_projects.pkl
│
├── 📁 analysis/
│   ├── domain_distribution.png
│   └── difficulty_distribution.png
│
├── 📁 src/
│   ├── preprocessing.py
│   ├── analyze_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── recommender.py
│
├── 📄 app.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 📊 Dataset

The project uses a CSV dataset containing information about different project topics.

The expected dataset file is:

```text
data/project_topics.csv
```

## Dataset Attributes

The dataset contains fields such as:

| Column        | Description               |
| ------------- | ------------------------- |
| `project_id`  | Unique project identifier |
| `title`       | Project title             |
| `domain`      | Project domain            |
| `description` | Project description       |
| `skills`      | Required technical skills |
| `career_goal` | Relevant career goal      |
| `difficulty`  | Project difficulty        |

---

# 🧹 Data Preprocessing

The preprocessing module is responsible for preparing raw project data for the recommendation system.

File:

```text
src/preprocessing.py
```

## Processing Steps

### 1️⃣ Load Dataset

The CSV file is loaded using Pandas.

```python
df = pd.read_csv(DATA_PATH)
```

### 2️⃣ Remove Duplicate Records

Duplicate projects are removed.

```python
df = df.drop_duplicates()
```

### 3️⃣ Handle Missing Values

Records missing important fields are removed.

### 4️⃣ Clean Text

Text is:

* Converted to string
* Stripped of unnecessary spaces
* Normalized
* Cleaned for consistent processing

### 5️⃣ Combine Features

Relevant fields are combined into:

```text
combined_features
```

Example:

```text
Machine Learning
+
Artificial Intelligence
+
Build a machine learning prediction system
+
Python, Pandas, Scikit-learn
+
AI Engineer
+
Intermediate
```

This combined representation is then used by the recommendation model.

---

# 🔢 TF-IDF Vectorization

TF-IDF stands for:

**Term Frequency-Inverse Document Frequency**

It is used to convert textual information into numerical vectors.

The system uses:

```python
TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)
```

## Term Frequency

Term Frequency measures how frequently a word appears in a document.

## Inverse Document Frequency

Inverse Document Frequency reduces the importance of words that appear frequently across many documents.

## TF-IDF

The combination allows the system to give greater importance to words that are more useful for distinguishing project topics.

---

# 📐 Cosine Similarity

After TF-IDF conversion, each project becomes a numerical vector.

The student's profile is also converted into a vector.

The system then calculates:

```text
Cosine Similarity
```

The similarity value generally ranges between:

```text
0 → No similarity
1 → Very high similarity
```

The formula is:

```text
Cosine Similarity =
(A · B) / (||A|| × ||B||)
```

Where:

* `A` = Student profile vector
* `B` = Project vector
* `A · B` = Dot product
* `||A||` = Magnitude of A
* `||B||` = Magnitude of B

Higher similarity means the project is more relevant to the student's profile.

---

# ⚙️ Installation

## 1️⃣ Clone or Download the Project

Place the project in your preferred directory.

Example:

```text
AI-Project-Topic-Recommender
```

---

## 2️⃣ Open Terminal

Navigate to the project directory:

```powershell
cd "AI-Project-Topic-Recommender"
```

---

## 3️⃣ Create Virtual Environment

```powershell
python -m venv .venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 📦 Requirements

The `requirements.txt` file should contain:

```text
pandas
numpy
scikit-learn
joblib
matplotlib
streamlit
```

You can install them using:

```powershell
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the files in the following order.

---

## 1️⃣ Preprocess Dataset

```powershell
python src/preprocessing.py
```

This performs:

* Dataset loading
* Cleaning
* Feature creation
* Dataset analysis

---

## 2️⃣ Analyze Dataset

```powershell
python src/analyze_data.py
```

This generates data analysis results and charts.

Generated files:

```text
analysis/
├── domain_distribution.png
└── difficulty_distribution.png
```

---

## 3️⃣ Train Recommendation Model

```powershell
python src/train_model.py
```

The training process creates:

```text
models/
├── tfidf_vectorizer.pkl
├── project_matrix.pkl
└── processed_projects.pkl
```

---

## 4️⃣ Evaluate Model

```powershell
python src/evaluate_model.py
```

The evaluation module calculates:

```text
Precision@5
```

---

## 5️⃣ Launch Streamlit Application

```powershell
streamlit run app.py
```

Streamlit will launch the web application in your browser.

---

# 📈 Data Analysis

The project includes exploratory data analysis to understand the dataset.

The analysis includes:

### 📊 Domain Distribution

Shows the number of projects available in each domain.

Example domains may include:

```text
Artificial Intelligence
Machine Learning
Data Science
Web Development
Cybersecurity
Computer Vision
Natural Language Processing
```

### 📊 Difficulty Distribution

Shows the number of projects classified as:

```text
Beginner
Intermediate
Advanced
```

### 📊 Missing Values

The system identifies missing information in each column.

### 🔁 Duplicate Detection

The system checks for duplicate project records.

---

# 🤖 Model Training

The training pipeline performs the following steps:

```text
1. Load dataset
       ↓
2. Clean dataset
       ↓
3. Create combined features
       ↓
4. Initialize TF-IDF
       ↓
5. Transform project text
       ↓
6. Generate feature matrix
       ↓
7. Save vectorizer
       ↓
8. Save project matrix
       ↓
9. Save processed dataset
```

The trained components are saved using `joblib`.

This allows the application to reuse the trained representation without rebuilding it every time.

---

# 📊 Model Evaluation

The project evaluates recommendation quality using:

## Precision@K

Precision@K measures how many of the top K recommendations are relevant according to the evaluation criterion.

For example:

```text
K = 5
```

means that the top 5 recommendations are evaluated.

The evaluation script can be executed using:

```powershell
python src/evaluate_model.py
```

Example output:

```text
============================================================
EVALUATION RESULTS
============================================================

Precision@5: 0.8000
Precision@5 Percentage: 80.00%

Projects evaluated: 20
Recommendations per query: 5

Evaluation completed successfully.
============================================================
```

> Note: The actual score depends on the dataset.

---

# 🖥️ Streamlit Application

The project includes an interactive web interface.

Users can enter:

### 💡 Interests

Example:

```text
Artificial Intelligence, Computer Vision, Generative AI
```

### 🧑‍💻 Technical Skills

Example:

```text
Python, OpenCV, Pandas, Machine Learning
```

### 🎯 Career Goal

Example:

```text
AI Engineer
```

### 🏢 Preferred Domain

Example:

```text
Artificial Intelligence
```

### 📊 Difficulty

Example:

```text
Intermediate
```

### 🔢 Number of Recommendations

Users can choose how many project recommendations they want.

---

# 🧪 Example Input

```text
Interests:
Artificial Intelligence, Computer Vision

Technical Skills:
Python, OpenCV, NumPy, Machine Learning

Career Goal:
AI Engineer

Preferred Domain:
Artificial Intelligence

Difficulty:
Intermediate

Recommendations:
5
```

---

# 🎯 Example Output

The application may return results such as:

```text
🎯 Recommended Projects

1. AI-Based Object Detection System

Domain:
Artificial Intelligence

Difficulty:
Intermediate

Required Skills:
Python, OpenCV, YOLO

Career Goal:
AI Engineer

Match Score:
91.25%
```

Additional relevant projects are displayed below according to their similarity scores.

---

# 🔐 Security Considerations

The current system does not require sensitive personal information.

Recommended security practices include:

* 🔒 Do not store unnecessary user information.
* 🔒 Do not expose API keys in source code.
* 🔒 Use environment variables for future API integrations.
* 🔒 Validate user input.
* 🔒 Restrict uploaded files to expected formats.
* 🔒 Keep dependencies updated.
* 🔒 Avoid executing arbitrary user-provided code.

If the project is deployed publicly, additional application security should be implemented.

---

# 📈 Scalability

The current system is suitable for small and medium-sized project datasets.

For larger datasets, the following improvements can be introduced:

### 🚀 Efficient Vector Search

Use vector databases or approximate nearest-neighbor algorithms.

Possible technologies:

* FAISS
* Elasticsearch
* Chroma
* Pinecone

### ⚡ Caching

Cache frequently requested recommendations.

### ☁️ Cloud Deployment

The application can be deployed using cloud platforms.

### 🗄️ Database Integration

Replace the CSV dataset with:

* PostgreSQL
* MySQL
* MongoDB

This would allow easier management of large numbers of projects.

---

# 🔧 Maintainability

The project follows a modular architecture.

Each major task is separated into its own file:

```text
preprocessing.py
        ↓
Data preparation

analyze_data.py
        ↓
Data analysis

train_model.py
        ↓
Model training

evaluate_model.py
        ↓
Model evaluation

recommender.py
        ↓
Recommendation engine

app.py
        ↓
User interface
```

This structure makes it easier to:

* 🛠️ Debug individual components
* ➕ Add new features
* 🔄 Replace algorithms
* 📊 Modify the dataset
* 🧪 Test individual modules
* 🚀 Deploy the application

---

# 🚀 Future Enhancements

Several improvements can be added in future versions.

## 🧠 Advanced NLP Models

Replace TF-IDF with transformer-based embeddings.

Possible models:

* BERT
* Sentence Transformers
* DistilBERT

This could improve semantic understanding.

---

## 👤 User Profiles

Allow students to create profiles and save:

* Skills
* Interests
* Career goals
* Previous projects
* Preferred technologies

---

## ⭐ Feedback System

Users could rate recommendations.

Example:

```text
👍 Relevant
👎 Not Relevant
⭐ 5/5
```

The feedback could later be used to improve recommendations.

---

## 📚 Learning Path Recommendation

The system could recommend:

```text
Skills → Courses → Projects → Career
```

For example:

```text
Python
   ↓
Machine Learning
   ↓
Computer Vision
   ↓
Object Detection Project
   ↓
AI Engineer
```

---

## 🔍 Semantic Search

Students could simply type:

```text
"I want a project related to AI and healthcare."
```

The system could automatically understand the intent and return suitable projects.

---

## 🌐 Real-Time Project Database

The application could eventually retrieve project ideas from online sources or an institutional project database.

---

## 📱 Mobile-Friendly Interface

The Streamlit interface can be improved for mobile and tablet users.

---

# 🐛 Troubleshooting

## ❌ Dataset Not Found

If you see:

```text
FileNotFoundError:
Dataset not found at:
data\project_topics.csv
```

make sure the dataset exists here:

```text
AI-Project-Topic-Recommender/
└── data/
    └── project_topics.csv
```

The project uses absolute path resolution based on the location of the Python files, so the application does not depend on the terminal's current directory.

---

## ❌ Model Files Missing

If Streamlit displays:

```text
Recommendation model files are missing.
```

run:

```powershell
python src/train_model.py
```

This should create:

```text
models/
├── tfidf_vectorizer.pkl
├── project_matrix.pkl
└── processed_projects.pkl
```

Then launch:

```powershell
streamlit run app.py
```

---

## ❌ ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'sklearn'
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

## ❌ Streamlit Not Recognized

Run:

```powershell
python -m streamlit run app.py
```

instead of:

```powershell
streamlit run app.py
```

---

## ❌ PowerShell Virtual Environment Error

If PowerShell blocks activation, you can run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 📚 Learning Outcomes

Through this project, the following technical skills are demonstrated:

* 🐍 Python Programming
* 📊 Data Analysis
* 🧹 Data Preprocessing
* 🧠 Natural Language Processing
* 🔢 TF-IDF Feature Extraction
* 📐 Cosine Similarity
* 🤖 Recommendation Systems
* 🧪 Model Evaluation
* 📈 Data Visualization
* 🖥️ Streamlit Development
* 💾 Model Serialization
* 🏗️ Modular Software Architecture
* 🔧 Error Handling
* 📂 File and Path Management

---

# 💡 Recommendation Approach

This project uses a:

## Content-Based Recommendation System

The system recommends projects based on the characteristics of the projects themselves.

The recommendation is generated using:

```text
Student Profile
       +
Project Features
       ↓
TF-IDF
       ↓
Numerical Vectors
       ↓
Cosine Similarity
       ↓
Similarity Ranking
       ↓
Top-N Projects
```

This approach does not require a large user-rating history, making it suitable for a newly developed recommendation platform.

---

# 📌 Advantages

### ✅ Simple and Efficient

The TF-IDF + Cosine Similarity approach is computationally efficient for small and medium datasets.

### ✅ Explainable

Recommendations are based on textual similarity, making the system easier to understand.

### ✅ No User History Required

The system can generate recommendations for new users immediately.

### ✅ Easy to Extend

Additional project attributes can be incorporated into the feature representation.

### ✅ Easy to Deploy

The application can be deployed using Streamlit-compatible hosting environments.

---

# ⚠️ Limitations

The current system has some limitations:

1. It relies heavily on the quality of the project dataset.
2. TF-IDF primarily captures lexical similarity rather than deep semantic meaning.
3. It does not currently learn from user feedback.
4. Recommendations depend on the available project topics.
5. The evaluation metric uses domain relevance as the primary relevance criterion.
6. A larger and more diverse dataset would improve recommendation quality.

---

# 🔮 Future System Architecture

A more advanced version could use:

```text
                 👤 Student
                     │
                     ▼
              📝 User Profile
                     │
                     ▼
             🧠 NLP Processing
                     │
                     ▼
          🔥 Sentence Embeddings
                     │
                     ▼
              📚 Vector Database
                     │
                     ▼
             🔍 Similarity Search
                     │
                     ▼
          🤖 Recommendation Engine
                     │
                     ▼
            🎯 Ranked Projects
                     │
                     ▼
                🖥️ Web App
                     │
                     ▼
               ⭐ User Feedback
                     │
                     └──────────────┐
                                    │
                                    ▼
                            🔄 Model Improvement
```

---

# 🏆 Project Highlights

This project demonstrates an end-to-end AI workflow:

```text
📊 Data Collection
       ↓
🧹 Data Cleaning
       ↓
🔍 Data Analysis
       ↓
📝 Feature Engineering
       ↓
🧠 NLP
       ↓
🔢 TF-IDF
       ↓
📐 Cosine Similarity
       ↓
🤖 Recommendation
       ↓
📊 Evaluation
       ↓
🖥️ Web Application
```

---

# 👨‍💻 Author

**Rahul Gupta**

🎓 B.Sc. (Hons) Computer Science
🏫 University of Delhi
💻 AI & Data Science Enthusiast

---

# 📄 License

This project is created for educational and internship purposes.

You are free to modify and improve the project for learning and academic use.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ and exploring the source code to understand how recommendation systems can be built using Python and machine learning.

---

## 🚀 Final Project Summary

The **AI Project Topic Recommender** is an end-to-end machine learning application designed to help students select suitable project topics.

It combines:

**Python + Pandas + Scikit-learn + NLP + TF-IDF + Cosine Similarity + Streamlit**

to transform student preferences into personalized project recommendations.

The project demonstrates the complete journey from **raw data analysis and preprocessing to model development, evaluation, and deployment through an interactive web interface.**

```
```
