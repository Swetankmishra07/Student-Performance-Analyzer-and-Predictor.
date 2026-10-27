"""
Model Training Module
======================
Step 5 & 6: Build, train, evaluate, and compare multiple ML models.

Models used:
1. Linear Regression - Simple baseline, assumes linear relationship
2. Decision Tree - Captures non-linear patterns, easy to interpret
3. Random Forest - Ensemble method, reduces overfitting, best accuracy

We predict the 'average_score' (regression) and 'performance_label' (classification).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import joblib
import os
import json


class ModelTrainer:
    """
    Trains and evaluates multiple ML models for:
    1. Regression: Predict average_score
    2. Classification: Predict performance_label
    """

    def __init__(self, output_dir="outputs/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, df, target_col, feature_cols=None, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        df : DataFrame with processed data
        target_col : Column to predict
        feature_cols : List of feature columns (auto-detected if None)
        test_size : Fraction of data for testing (default 20%)
        random_state : Seed for reproducibility
        """
        if feature_cols is None:
            # Auto-select: all columns except target and score-related columns
            exclude = [target_col, 'math_score', 'reading_score', 'writing_score',
                        'average_score', 'total_score', 'performance_label']
            feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"📦 Data Split:")
        print(f"   Features: {feature_cols}")
        print(f"   Target: {target_col}")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """
        Train regression models to predict average_score.
        
        Models:
        1. Linear Regression - Baseline. Finds the best straight line.
        2. Decision Tree Regressor - Splits data into regions. Can overfit.
        3. Random Forest Regressor - Many trees voting together. Best accuracy.
        """
        print("\n" + "=" * 60)
        print("📈 TRAINING REGRESSION MODELS")
        print("=" * 60)

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=8, min_samples_split=10, random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\n🔄 Training: {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            results[name] = {
                'model': model,
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3),
                'R2_Score': round(r2, 4),
                'CV_Mean_R2': round(cv_scores.mean(), 4),
                'CV_Std_R2': round(cv_scores.std(), 4)
            }

            print(f"   ✅ {name} Results:")
            print(f"      MAE  = {mae:.3f} (avg error in score points)")
            print(f"      RMSE = {rmse:.3f} (penalizes large errors)")
            print(f"      R²   = {r2:.4f} (1.0 = perfect, higher is better)")
            print(f"      CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Find best model
        best_name = max(results, key=lambda k: results[k]['R2_Score'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.results['regression'] = results

        print(f"\n🏆 Best Regression Model: {best_name} (R² = {results[best_name]['R2_Score']:.4f})")
        return results

    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """
        Train classification models to predict performance_label.
        
        Models:
        1. Logistic Regression - Simple, fast, good for linearly separable data
        2. Decision Tree Classifier - Rule-based, interpretable
        3. Random Forest Classifier - Ensemble, high accuracy
        """
        print("\n" + "=" * 60)
        print("🏷️  TRAINING CLASSIFICATION MODELS")
        print("=" * 60)

        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=10, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\n🔄 Training: {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            results[name] = {
                'model': model,
                'accuracy': round(acc, 4),
                'CV_Mean_Accuracy': round(cv_scores.mean(), 4),
                'CV_Std_Accuracy': round(cv_scores.std(), 4),
                'classification_report': classification_report(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

            print(f"   ✅ {name} Results:")
            print(f"      Accuracy = {acc:.4f}")
            print(f"      CV Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"\n   Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        best_name = max(results, key=lambda k: results[k]['accuracy'])
        self.results['classification'] = results

        print(f"\n🏆 Best Classification Model: {best_name} (Accuracy = {results[best_name]['accuracy']:.4f})")
        return results

    def get_feature_importance(self, model, feature_names):
        """Extract and display feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            return None

        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\n📊 Feature Importance:")
        for _, row in feat_imp.iterrows():
            bar = '█' * int(row['Importance'] * 50)
            print(f"   {row['Feature']:30s} {row['Importance']:.4f} {bar}")

        return feat_imp

    def save_model(self, model, model_name, feature_names):
        """Save model and metadata for deployment."""
        model_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)

        meta_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_metadata.json")
        metadata = {
            'model_name': model_name,
            'feature_names': feature_names,
            'model_path': model_path
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Model saved: {model_path}")
        print(f"💾 Metadata saved: {meta_path}")
        return model_path

    def comparison_table(self, results_dict):
        """Create a comparison table of all models."""
        rows = []
        for name, metrics in results_dict.items():
            row = {'Model': name}
            for k, v in metrics.items():
                if k != 'model' and k != 'classification_report' and k != 'confusion_matrix':
                    row[k] = v
            rows.append(row)
        
        df_results = pd.DataFrame(rows).set_index('Model')
        print("\n📋 Model Comparison:")
        print(df_results.to_string())
        return df_results


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")
    df = pd.read_csv(data_path)

    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.full_pipeline(df)

    # Train
    trainer = ModelTrainer(
        output_dir=os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    )

    # Regression
    feature_cols = ['gender', 'race_ethnicity', 'parental_level_of_education',
                    'lunch', 'test_preparation_course', 'study_hours_per_week', 'attendance_rate']
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(
        df_processed, target_col='average_score', feature_cols=feature_cols
    )
    reg_results = trainer.train_regression_models(X_train, X_test, y_train, y_test)
    trainer.comparison_table(reg_results)
    trainer.get_feature_importance(reg_results['Random Forest']['model'], features)

    # Save best regression model
    trainer.save_model(trainer.best_model, trainer.best_model_name + "_regression", features)

    # Classification
    X_train_c, X_test_c, y_train_c, y_test_c, features_c = trainer.prepare_data(
        df_processed, target_col='performance_label', feature_cols=feature_cols
    )
    cls_results = trainer.train_classification_models(X_train_c, X_test_c, y_train_c, y_test_c)
    trainer.comparison_table(cls_results)
