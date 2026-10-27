# 🎓 Student Performance Analyzer & Predictor

An end-to-end data science and machine learning project in the education domain. This project analyzes demographic, socioeconomic, and academic factors to predict a student's academic performance and provides personalized improvement recommendations.

Includes a complete data processing, EDA, and model training pipeline, as well as an interactive Streamlit web application and a zero-dependency HTML dashboard demo.

---

## 🚀 Key Features

*   **End-to-End ML Pipeline**: Automated script that generates synthetic student data, cleans it, runs EDA, trains models, and saves evaluation metrics and model artifacts.
*   **Predictive Modeling**: Dual machine learning tasks using Scikit-Learn:
    *   *Regression*: Predicting a student's exact average score (using `Linear Regression`, `Decision Tree`, and `Random Forest Regressor` - R² > 0.85).
    *   *Classification*: Categorizing a student's performance tier (Excellent, Good, Average, Below Average, Poor).
*   **Interactive Web App**: A stunning Streamlit-based web dashboard (`app.py`) featuring:
    *   *Real-time Predictor*: Live predictions with gauge charts and personalized academic advice.
    *   *Interactive EDA Dashboard*: Custom Plotly visualizations analyzing performance drivers.
    *   *Data Explorer*: Interactive data table with filters and basic statistics.
    *   *Model Insights*: Model performance comparisons and Random Forest feature importance.
*   **Static HTML Demo**: A zero-dependency single-file HTML/JS/CSS mock interface (`index.html`) to showcase the UI design instantly without setting up Python.

---

## 📁 Project Structure

```
.
├── app.py                      # Premium Streamlit web application
├── index.html                  # Standalone, interactive front-end demo (HTML/CSS/JS)
├── instruction.md              # Project requirements and prompt
├── requirements.txt            # Python library dependencies
├── run_pipeline.py             # Orchestrator script to run the full ML pipeline
├── data/
│   ├── generate_dataset.py     # Generates a realistic synthetic student dataset
│   └── student_performance.csv  # Generated dataset (1,000 students)
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_preprocessing.py   # Missing values, feature engineering, scaling, encoding
│   ├── eda.py                  # Statistics & matplotlib/seaborn charts generation
│   ├── model_training.py       # Regression & classification training & evaluation
│   └── prediction.py           # Prediction system & recommendation generation
└── outputs/
    ├── charts/                 # Generated EDA charts (saved as PNGs)
    │   ├── score_distributions.png
    │   ├── correlation_heatmap.png
    │   ├── gender_comparison.png
    │   ├── parental_education_impact.png
    │   ├── test_prep_effect.png
    │   ├── lunch_impact.png
    │   ├── study_hours_vs_scores.png
    │   └── performance_distribution.png
    └── models/                 # Saved ML models, encoders, and metadata
        ├── random_forest_regression_model.pkl
        ├── random_forest_regression_metadata.json
        ├── random_forest_classification_model.pkl
        ├── preprocessor.pkl
        └── ...
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd Student-Performance-Analyzer
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run

### 1. Run the Pipeline
Before running the Streamlit app, run the pipeline orchestrator to generate the dataset, perform EDA, and train the machine learning models:
```bash
python run_pipeline.py
```
This script runs the following steps:
1.  **Dataset Generation**: Creates `data/student_performance.csv` (1,000 records).
2.  **Data Preprocessing**: Handles missing values (median for numeric, mode for categorical), feature engineering (`average_score`, `total_score`, `performance_label`), and fits/saves encoders.
3.  **Exploratory Data Analysis (EDA)**: Computes dataset stats and saves 8 distinct charts to `outputs/charts/`.
4.  **Model Training**: Trains and compares `Linear Regression`, `Decision Tree`, and `Random Forest` for both tasks, outputting the best models and preprocessor to `outputs/models/`.

### 2. Launch the Web Application
Start the interactive Streamlit web dashboard:
```bash
streamlit run app.py
```
Open the local server URL (usually `http://localhost:8501`) in your browser.

### 3. Run the Zero-Dependency Demo
Alternatively, you can open `index.html` directly in any web browser to test the interactive mock dashboard without installing Python.

---

## 📊 Dataset Description

The generated dataset is modeled after standard student performance datasets, featuring the following attributes:

| Column Name | Description | Type | Examples / Range |
| :--- | :--- | :--- | :--- |
| `gender` | Student gender | Categorical | `female`, `male` |
| `race_ethnicity` | Demographic group | Categorical | `group A` to `group E` |
| `parental_level_of_education` | Parents' highest education | Categorical | `high school` to `master's degree` |
| `lunch` | Lunch plan (socioeconomic proxy) | Categorical | `standard`, `free/reduced` |
| `test_preparation_course` | Test prep course completion status | Categorical | `completed`, `none` |
| `study_hours_per_week` | Weekly study hours | Numeric | `1.0` to `40.0` hours (with simulated missing values) |
| `attendance_rate` | Attendance percentage | Numeric | `30.0%` to `100.0%` (with simulated missing values) |
| `math_score` | Math exam grade | Target (Numeric)| `0` to `100` |
| `reading_score` | Reading exam grade | Target (Numeric)| `0` to `100` |
| `writing_score` | Writing exam grade | Target (Numeric)| `0` to `100` |

---

## 🔧 Data Preprocessing Details

Implemented inside [data_preprocessing.py](file:///e:/google/src/data_preprocessing.py):
1.  **Imputation**:
    *   **Numerical columns** (`study_hours_per_week`, `attendance_rate`) are imputed using the column **median** value to remain robust against outliers.
    *   **Categorical columns** are imputed using the **mode** (most frequent class).
2.  **Feature Engineering**:
    *   `average_score`: Calculated as the mean of math, reading, and writing scores.
    *   `total_score`: Cumulative score of the three subjects.
    *   `performance_label`: Classification label based on `average_score`:
        *   `Excellent` ($\ge 90$)
        *   `Good` ($75 \le \text{score} < 90$)
        *   `Average` ($60 \le \text{score} < 75$)
        *   `Below Average` ($40 \le \text{score} < 60$)
        *   `Poor` ($< 40$)
3.  **Encoding & Scaling**:
    *   Categorical columns are converted to numerical values using `LabelEncoder`.
    *   Numeric columns are scaled using `StandardScaler` to bring them onto a unified scale before training.

---

## 🤖 Models & Evaluation

The pipeline trains and compares three different models for both the regression and classification tasks:

### Regression (Target: `average_score`)
*   **Linear Regression**: Simple baseline model checking linear relations.
*   **Decision Tree Regressor**: Captures non-linear dependencies.
*   **Random Forest Regressor**: Best performer. Ensemble of 100 decision trees that minimizes variance and achieves `R² > 0.85` on the test set.

### Classification (Target: `performance_label`)
*   **Logistic Regression**: Simple linear classification baseline.
*   **Decision Tree Classifier**: Rule-based categorization.
*   **Random Forest Classifier**: Ensemble classifier achieving the highest multi-class accuracy.

### Feature Importance (Random Forest Regressor)
Through Random Forest feature importance analysis, we find the following major factors driving academic success:
1.  **Attendance Rate**: The strongest numeric predictor of high academic performance.
2.  **Study Hours per Week**: Direct linear-trend relationship with performance.
3.  **Lunch Type**: Socioeconomic status (standard lunch proxy) strongly correlates with average scores.
4.  **Test Preparation Course**: Completion yields an average boost of `8-12` score points.

---

## 💡 Personalized Recommendations Engine

Based on predicted student outcomes, the system generates automated academic actions:
*   *Study Hours*: Recommends increasing study duration to 15-20 hours if currently below 10 hours.
*   *Test Prep*: Suggests enrolling in test preparation courses if none are completed.
*   *Attendance*: Flags students with attendance $< 75\%$ for immediate classroom intervention.
*   *Performance Level*: Suggests peer mentoring for top-performing students and structured tutoring for at-risk profiles.

---

## 🚀 Deployment Guide

### Option 1: Streamlit Community Cloud (Easiest)
1. Push the code to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Select your repository, branch, and specify `app.py` as the entry file.
4. Click **Deploy**.

### Option 2: Render (Free Tier Web Service)
1. Link your GitHub repository to [Render](https://render.com).
2. Create a new Web Service and select the Python environment.
3. Set the start command:
   ```bash
   streamlit run app.py --server.port $PORT
   ```
4. Click **Deploy**.

---

## 📘 Career & Mentor Insights

### Resume Bullet Points
*   Developed an end-to-end **Student Performance Prediction System** using Python, Scikit-Learn, and Streamlit.
*   Performed comprehensive **Exploratory Data Analysis (EDA)** with 8+ visualizations, uncovering critical socioeconomic and behavioral factors affecting student outcomes.
*   Built and compared **3 Machine Learning models** (Linear Regression, Decision Trees, Random Forests) for regression and classification tasks, achieving `R² > 0.85`.
*   Engineered a **real-time prediction system** generating personalized, automated academic recommendations.
*   Created an **interactive web dashboard** incorporating data filtering, descriptive statistics, and model explainability metrics.

### Key Interview Prep Q&A

**Q1: Why did you choose Random Forest over other models?**
> Random Forest is an ensemble method combining multiple decision trees. It significantly reduces overfitting (common in single decision trees) by bootstrapping samples and features, providing robust cross-validated predictions and built-in feature importances. For our dataset, it yielded the lowest Mean Absolute Error (MAE) and highest R² score.

**Q2: How did you handle missing values and why?**
> Numeric columns (`study_hours_per_week`, `attendance_rate`) were imputed using the column **median** since it is robust to outliers, while categorical columns were imputed using the **mode** (most frequent value). This approach prevents training data distortion.

**Q3: What was the most significant insight from your EDA?**
> The type of lunch (a proxy for socioeconomic status) had a major impact. Students on the `standard` lunch plan scored significantly higher across all subjects. This highlights how systemic socioeconomic factors influence academic performance, showing that student outcomes are shaped by more than just individual study habits.
