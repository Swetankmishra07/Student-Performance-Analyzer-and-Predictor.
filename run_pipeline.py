"""
Main Pipeline Runner
=====================
Orchestrates the full data science workflow:
1. Generate dataset
2. Preprocess data
3. Run EDA
4. Train models
5. Save artifacts for the Streamlit app

Run this script first to set up everything before launching the UI.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import numpy as np

from data.generate_dataset import generate_student_data
from src.data_preprocessing import DataPreprocessor
from src.eda import EDAAnalyzer
from src.model_training import ModelTrainer


def run_pipeline():
    """Run the complete Student Performance Analyzer pipeline."""
    
    print("\n" + "🎓" * 30)
    print("  STUDENT PERFORMANCE ANALYZER & PREDICTOR")
    print("  Complete End-to-End Data Science Pipeline")
    print("🎓" * 30 + "\n")

    # =============================================
    # STEP 1: Generate / Load Dataset
    # =============================================
    print("\n" + "=" * 70)
    print("📁 STEP 1: DATASET GENERATION")
    print("=" * 70)

    data_path = os.path.join(PROJECT_ROOT, "data", "student_performance.csv")
    
    if not os.path.exists(data_path):
        print("Generating synthetic student dataset...")
        df = generate_student_data(n_students=1000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset saved: {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {data_path}")

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample Data:")
    print(df.head(5).to_string(index=False))

    # =============================================
    # STEP 2: DATA PREPROCESSING
    # =============================================
    print("\n\n" + "=" * 70)
    print("🔧 STEP 2: DATA PREPROCESSING")
    print("=" * 70)

    preprocessor = DataPreprocessor()
    df_processed = preprocessor.full_pipeline(df, fit=True)

    # Save the preprocessor for the Streamlit app
    preprocessor_path = os.path.join(PROJECT_ROOT, "outputs", "models", "preprocessor.pkl")
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    preprocessor.save_preprocessor(preprocessor_path)

    print(f"\n   Processed Shape: {df_processed.shape}")
    print(f"\n   Processed Sample:")
    print(df_processed.head(5).to_string(index=False))

    # =============================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # =============================================
    print("\n\n" + "=" * 70)
    print("🔬 STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    charts_dir = os.path.join(PROJECT_ROOT, "outputs", "charts")
    analyzer = EDAAnalyzer(df, output_dir=charts_dir)  # Use raw data for EDA
    analyzer.run_full_eda()

    # =============================================
    # STEP 4: MODEL TRAINING
    # =============================================
    print("\n\n" + "=" * 70)
    print("🤖 STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    trainer = ModelTrainer(output_dir=models_dir)

    # Define feature columns
    feature_cols = [
        'gender', 'race_ethnicity', 'parental_level_of_education',
        'lunch', 'test_preparation_course', 'study_hours_per_week',
        'attendance_rate'
    ]

    # --- REGRESSION: Predict average score ---
    print("\n" + "-" * 50)
    print("📈 REGRESSION TASK: Predict Average Score")
    print("-" * 50)

    X_train, X_test, y_train, y_test, features = trainer.prepare_data(
        df_processed, target_col='average_score', feature_cols=feature_cols
    )
    reg_results = trainer.train_regression_models(X_train, X_test, y_train, y_test)
    trainer.comparison_table(reg_results)

    # Feature importance for best model
    best_reg_model = reg_results['Random Forest']['model']
    trainer.get_feature_importance(best_reg_model, features)

    # Save best regression model
    trainer.save_model(best_reg_model, "random_forest_regression", features)

    # --- CLASSIFICATION: Predict performance label ---
    print("\n" + "-" * 50)
    print("🏷️  CLASSIFICATION TASK: Predict Performance Label")
    print("-" * 50)

    X_train_c, X_test_c, y_train_c, y_test_c, features_c = trainer.prepare_data(
        df_processed, target_col='performance_label', feature_cols=feature_cols
    )
    cls_results = trainer.train_classification_models(X_train_c, X_test_c, y_train_c, y_test_c)
    trainer.comparison_table(cls_results)

    # Save best classification model
    best_cls_name = max(cls_results, key=lambda k: cls_results[k]['accuracy'])
    trainer.save_model(cls_results[best_cls_name]['model'], "random_forest_classification", features_c)

    # =============================================
    # SUMMARY
    # =============================================
    print("\n\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"""
    📊 Files Generated:
    ├── data/student_performance.csv          (Dataset)
    ├── outputs/
    │   ├── charts/                            (EDA Visualizations)
    │   │   ├── score_distributions.png
    │   │   ├── correlation_heatmap.png
    │   │   ├── gender_comparison.png
    │   │   ├── parental_education_impact.png
    │   │   ├── test_prep_effect.png
    │   │   ├── lunch_impact.png
    │   │   ├── study_hours_vs_scores.png
    │   │   └── performance_distribution.png
    │   └── models/                            (Trained Models)
    │       ├── random_forest_regression_model.pkl
    │       ├── random_forest_classification_model.pkl
    │       └── preprocessor.pkl

    🚀 Next Step: Launch the Streamlit UI
       Run: streamlit run app.py
    """)


if __name__ == "__main__":
    run_pipeline()
