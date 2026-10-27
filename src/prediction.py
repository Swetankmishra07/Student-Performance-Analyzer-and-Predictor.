"""
Prediction Module
==================
Step 7: Build a prediction system where users input data and get predictions.

This module loads a trained model and preprocessor,
accepts new student data, and returns predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json


class StudentPredictor:
    """
    Prediction system for student performance.
    Loads a trained model and preprocessor to make predictions on new data.
    """

    def __init__(self, model_dir="outputs/models"):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor_data = None
        self.feature_names = None
        self.label_encoders = None

    def load_model(self, model_filename):
        """Load a trained model from disk."""
        model_path = os.path.join(self.model_dir, model_filename)
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")

        # Try to load metadata
        meta_name = model_filename.replace('_model.pkl', '_metadata.json')
        meta_path = os.path.join(self.model_dir, meta_name)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_names = meta.get('feature_names', [])
            print(f"   Features: {self.feature_names}")

    def load_preprocessor(self, preprocessor_path):
        """Load the fitted preprocessor."""
        self.preprocessor_data = joblib.load(preprocessor_path)
        self.label_encoders = self.preprocessor_data.get('label_encoders', {})
        print(f"✅ Preprocessor loaded: {preprocessor_path}")

    def encode_input(self, input_data):
        """
        Encode user input using the saved label encoders.
        
        Parameters:
        -----------
        input_data : dict with raw user inputs
            Example: {
                'gender': 'female',
                'race_ethnicity': 'group C',
                'parental_level_of_education': "bachelor's degree",
                'lunch': 'standard',
                'test_preparation_course': 'completed',
                'study_hours_per_week': 20.0,
                'attendance_rate': 90.0
            }
        """
        encoded = {}
        for key, value in input_data.items():
            if key in self.label_encoders:
                le = self.label_encoders[key]
                if value in le.classes_:
                    encoded[key] = le.transform([value])[0]
                else:
                    print(f"⚠️ Unknown value '{value}' for '{key}', using 0")
                    encoded[key] = 0
            else:
                encoded[key] = value
        return encoded

    def predict(self, input_data):
        """
        Make a prediction for a single student.
        
        Parameters:
        -----------
        input_data : dict with student information
        
        Returns:
        --------
        prediction : float (predicted average score)
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")

        # Encode categorical inputs
        if self.label_encoders:
            encoded = self.encode_input(input_data)
        else:
            encoded = input_data

        # Create DataFrame with correct feature order
        if self.feature_names:
            df_input = pd.DataFrame([encoded])[self.feature_names]
        else:
            df_input = pd.DataFrame([encoded])

        # Make prediction
        prediction = self.model.predict(df_input)[0]
        return prediction

    def predict_with_details(self, input_data):
        """
        Make a prediction and return detailed results.
        
        Returns:
        --------
        dict with prediction, performance label, and recommendations
        """
        predicted_score = self.predict(input_data)

        # Determine performance label
        if predicted_score >= 90:
            label = "Excellent"
            emoji = "🌟"
            color = "green"
        elif predicted_score >= 75:
            label = "Good"
            emoji = "👍"
            color = "blue"
        elif predicted_score >= 60:
            label = "Average"
            emoji = "📊"
            color = "orange"
        elif predicted_score >= 40:
            label = "Below Average"
            emoji = "⚠️"
            color = "red"
        else:
            label = "Poor"
            emoji = "🚨"
            color = "darkred"

        # Generate recommendations
        recommendations = self._generate_recommendations(input_data, predicted_score)

        result = {
            'predicted_score': round(predicted_score, 1),
            'performance_label': label,
            'emoji': emoji,
            'color': color,
            'recommendations': recommendations,
            'input_summary': input_data
        }

        return result

    def _generate_recommendations(self, input_data, score):
        """Generate personalized recommendations based on input and prediction."""
        recs = []

        # Study hours recommendation
        study_hrs = input_data.get('study_hours_per_week', 0)
        if study_hrs < 10:
            recs.append("📚 Increase study hours to at least 15-20 hours per week for better results.")
        elif study_hrs < 20:
            recs.append("📖 Good study routine! Consider adding 5 more hours for maximum improvement.")

        # Test prep recommendation
        if input_data.get('test_preparation_course') == 'none':
            recs.append("📝 Completing a test preparation course can boost scores by 8-12 points on average.")

        # Attendance recommendation
        attendance = input_data.get('attendance_rate', 0)
        if attendance < 75:
            recs.append("🏫 Attendance is below 75%. Regular attendance strongly correlates with better performance.")
        elif attendance < 90:
            recs.append("🎯 Try to maintain attendance above 90% for optimal learning outcomes.")

        # Score-based recommendations
        if score < 50:
            recs.append("💡 Consider seeking additional tutoring or joining study groups.")
            recs.append("🎯 Focus on building strong fundamentals in core subjects.")
        elif score < 70:
            recs.append("📈 You're on the right track! Practice with mock tests to identify weak areas.")
        elif score >= 90:
            recs.append("🏆 Excellent performance! Consider mentoring other students or exploring advanced topics.")

        return recs


def interactive_prediction():
    """Command-line interactive prediction system."""
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE PREDICTOR")
    print("=" * 60)
    print("\nEnter student details to predict their performance:\n")

    # Collect inputs
    gender = input("Gender (male/female): ").strip().lower()
    race = input("Race/Ethnicity (group A/B/C/D/E): ").strip().lower()
    parent_edu = input("Parental Education (some high school / high school / some college / "
                       "associate's degree / bachelor's degree / master's degree): ").strip().lower()
    lunch = input("Lunch type (standard / free/reduced): ").strip().lower()
    test_prep = input("Test prep course (completed / none): ").strip().lower()
    study_hours = float(input("Study hours per week: ").strip())
    attendance = float(input("Attendance rate (%): ").strip())

    input_data = {
        'gender': gender,
        'race_ethnicity': race,
        'parental_level_of_education': parent_edu,
        'lunch': lunch,
        'test_preparation_course': test_prep,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance
    }

    # Load and predict
    predictor = StudentPredictor()
    # Try to find model files
    model_files = [f for f in os.listdir(predictor.model_dir) if f.endswith('_model.pkl')]
    if model_files:
        predictor.load_model(model_files[0])
        preprocessor_path = os.path.join(predictor.model_dir, '..', 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            predictor.load_preprocessor(preprocessor_path)

        result = predictor.predict_with_details(input_data)

        print(f"\n{'=' * 60}")
        print(f"📊 PREDICTION RESULTS")
        print(f"{'=' * 60}")
        print(f"\n   {result['emoji']} Predicted Average Score: {result['predicted_score']}")
        print(f"   📝 Performance Label: {result['performance_label']}")
        print(f"\n   💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"      {rec}")
    else:
        print("❌ No trained model found. Please run model_training.py first.")


if __name__ == "__main__":
    interactive_prediction()
