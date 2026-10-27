"""
Data Preprocessing Module
==========================
Step 3: Handle missing values, encode categorical variables, and scale features.

This module provides a clean pipeline to transform raw student data
into a format ready for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataPreprocessor:
    """
    Handles all data preprocessing steps:
    1. Missing value imputation
    2. Feature engineering (average_score, performance_label)
    3. Categorical encoding
    4. Feature scaling
    """

    def __init__(self):
        self.label_encoders = {}   # Stores fitted LabelEncoders per column
        self.scaler = StandardScaler()
        self.numeric_cols = ['study_hours_per_week', 'attendance_rate']
        self.categorical_cols = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course'
        ]

    def handle_missing_values(self, df):
        """
        Fill missing values:
        - Numeric columns → filled with median (robust to outliers)
        - Categorical columns → filled with mode (most frequent value)
        """
        df = df.copy()
        print("🔧 Handling missing values...")
        print(f"   Before: {df.isnull().sum().sum()} total missing values")

        for col in self.numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   → {col}: filled with median = {median_val:.1f}")

        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"   → {col}: filled with mode = '{mode_val}'")

        print(f"   After: {df.isnull().sum().sum()} total missing values ✅")
        return df

    def engineer_features(self, df):
        """
        Create new meaningful features:
        - average_score: Mean of math, reading, and writing scores
        - total_score: Sum of all three scores
        - performance_label: Categorical label based on average score
        """
        df = df.copy()
        print("\n🛠️  Engineering features...")

        # Average and total scores
        df['average_score'] = ((df['math_score'] + df['reading_score'] + df['writing_score']) / 3).round(1)
        df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']

        # Performance label based on average score
        def label_performance(avg):
            if avg >= 90:
                return 'Excellent'
            elif avg >= 75:
                return 'Good'
            elif avg >= 60:
                return 'Average'
            elif avg >= 40:
                return 'Below Average'
            else:
                return 'Poor'

        df['performance_label'] = df['average_score'].apply(label_performance)
        print("   → Created: average_score, total_score, performance_label ✅")
        return df

    def encode_categorical(self, df, fit=True):
        """
        Encode categorical variables using LabelEncoder.
        - fit=True: Fit and transform (for training data)
        - fit=False: Transform only (for new/test data)
        """
        df = df.copy()
        print("\n🔤 Encoding categorical variables...")

        cols_to_encode = self.categorical_cols.copy()
        if 'performance_label' in df.columns:
            cols_to_encode.append('performance_label')

        for col in cols_to_encode:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                    print(f"   → {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels gracefully
                        df[col] = df[col].map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

        print("   Encoding complete ✅")
        return df

    def scale_features(self, df, feature_cols, fit=True):
        """
        Scale numeric features using StandardScaler.
        - fit=True: Fit and transform (training)
        - fit=False: Transform only (prediction)
        """
        df = df.copy()
        print("\n📏 Scaling features...")

        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        print(f"   Scaled columns: {feature_cols} ✅")
        return df

    def save_preprocessor(self, path):
        """Save the fitted preprocessor (encoders + scaler) for deployment."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols
        }, path)
        print(f"\n💾 Preprocessor saved to: {path}")

    def load_preprocessor(self, path):
        """Load a previously fitted preprocessor."""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.numeric_cols = data['numeric_cols']
        self.categorical_cols = data['categorical_cols']
        print(f"📂 Preprocessor loaded from: {path}")

    def full_pipeline(self, df, fit=True):
        """
        Run the complete preprocessing pipeline:
        1. Handle missing values
        2. Engineer features
        3. Encode categoricals
        """
        print("=" * 60)
        print("🚀 RUNNING FULL PREPROCESSING PIPELINE")
        print("=" * 60)
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit=fit)
        # Safety net: fill any remaining NaN with column median/0
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"\n⚠️  {remaining_nan} NaN values remain — filling with column median...")
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 60)
        return df


if __name__ == "__main__":
    # Quick test
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")

    preprocessor = DataPreprocessor()
    df_processed = preprocessor.full_pipeline(df)
    print(f"\nProcessed dataset: {df_processed.shape}")
    print(df_processed.head())
