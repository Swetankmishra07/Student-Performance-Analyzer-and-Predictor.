"""
Dataset Generator for Student Performance Analyzer
====================================================
This script generates a realistic synthetic dataset modeled after 
the famous 'Students Performance in Exams' dataset from Kaggle.

Columns:
- gender: Male / Female
- race_ethnicity: Group A through E
- parental_level_of_education: Education level of parents
- lunch: Standard or free/reduced lunch program
- test_preparation_course: Completed or not
- study_hours_per_week: Weekly study hours (numeric)
- attendance_rate: Percentage attendance (numeric)
- math_score: Score in math (0-100)
- reading_score: Score in reading (0-100)
- writing_score: Score in writing (0-100)
"""

import pandas as pd
import numpy as np
import os

def generate_student_data(n_students=1000, seed=42):
    """Generate a realistic student performance dataset."""
    np.random.seed(seed)

    # --- Categorical Features ---
    gender = np.random.choice(['male', 'female'], size=n_students, p=[0.48, 0.52])

    race_ethnicity = np.random.choice(
        ['group A', 'group B', 'group C', 'group D', 'group E'],
        size=n_students,
        p=[0.09, 0.19, 0.32, 0.26, 0.14]
    )

    parental_education_levels = [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    parental_level_of_education = np.random.choice(
        parental_education_levels,
        size=n_students,
        p=[0.12, 0.20, 0.23, 0.22, 0.13, 0.10]
    )

    lunch = np.random.choice(
        ['standard', 'free/reduced'],
        size=n_students,
        p=[0.645, 0.355]
    )

    test_preparation_course = np.random.choice(
        ['completed', 'none'],
        size=n_students,
        p=[0.36, 0.64]
    )

    # --- Numeric Features ---
    study_hours_per_week = np.clip(np.random.normal(15, 7, n_students), 1, 40).round(1)
    attendance_rate = np.clip(np.random.normal(82, 12, n_students), 30, 100).round(1)

    # --- Generate Scores with Realistic Correlations ---
    # Base score influenced by preparation, lunch, parental education, study hours
    base = np.random.normal(65, 12, n_students)

    # Boost from test prep
    prep_boost = np.where(test_preparation_course == 'completed', 
                          np.random.normal(8, 2, n_students), 0)

    # Boost from standard lunch (proxy for socioeconomic status)
    lunch_boost = np.where(lunch == 'standard',
                           np.random.normal(5, 2, n_students), 0)

    # Boost from parental education
    edu_map = {
        "some high school": 0, "high school": 2, "some college": 4,
        "associate's degree": 6, "bachelor's degree": 9, "master's degree": 12
    }
    edu_boost = np.array([edu_map[e] for e in parental_level_of_education])

    # Boost from study hours and attendance
    study_boost = (study_hours_per_week - 15) * 0.5
    attendance_boost = (attendance_rate - 82) * 0.3

    # Compute scores with some noise
    math_score = np.clip(
        base + prep_boost + lunch_boost + edu_boost + study_boost + 
        attendance_boost + np.random.normal(0, 5, n_students), 0, 100
    ).round(0).astype(int)

    reading_score = np.clip(
        base + prep_boost * 1.1 + lunch_boost + edu_boost * 1.1 + 
        study_boost * 0.8 + attendance_boost + np.random.normal(3, 5, n_students), 0, 100
    ).round(0).astype(int)

    writing_score = np.clip(
        base + prep_boost * 1.2 + lunch_boost + edu_boost * 1.2 + 
        study_boost * 0.7 + attendance_boost + np.random.normal(2, 5, n_students), 0, 100
    ).round(0).astype(int)

    # --- Introduce some missing values (realistic) ---
    # About 2% missing in study_hours, 1.5% in attendance
    study_hours_per_week = study_hours_per_week.astype(float)
    attendance_rate = attendance_rate.astype(float)
    
    missing_study = np.random.choice(n_students, size=int(n_students * 0.02), replace=False)
    missing_attend = np.random.choice(n_students, size=int(n_students * 0.015), replace=False)
    study_hours_per_week[missing_study] = np.nan
    attendance_rate[missing_attend] = np.nan

    # Build DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'race_ethnicity': race_ethnicity,
        'parental_level_of_education': parental_level_of_education,
        'lunch': lunch,
        'test_preparation_course': test_preparation_course,
        'study_hours_per_week': study_hours_per_week,
        'attendance_rate': attendance_rate,
        'math_score': math_score,
        'reading_score': reading_score,
        'writing_score': writing_score
    })

    return df


if __name__ == "__main__":
    print("🎓 Generating Student Performance Dataset...")
    df = generate_student_data(n_students=1000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "student_performance.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n📊 Sample data:")
    print(df.head())
