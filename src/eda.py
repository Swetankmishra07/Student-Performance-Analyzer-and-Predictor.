"""
Exploratory Data Analysis (EDA) Module
========================================
Step 4: Generate insights, visualizations, and identify key factors
affecting student performance.

This module creates publication-quality charts and saves them to the
outputs/charts/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120


class EDAAnalyzer:
    """Performs comprehensive Exploratory Data Analysis on student data."""

    def __init__(self, df, output_dir="outputs/charts"):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📊 EDA Analyzer initialized | {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Charts will be saved to: {output_dir}/\n")

    def basic_statistics(self):
        """Print basic dataset statistics."""
        print("=" * 60)
        print("📋 BASIC STATISTICS")
        print("=" * 60)
        print(f"\nShape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nDescriptive Statistics:\n{self.df.describe().round(2)}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        return self.df.describe()

    def plot_score_distributions(self):
        """Plot distribution of math, reading, and writing scores."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        subjects = ['math_score', 'reading_score', 'writing_score']
        titles = ['Math Score', 'Reading Score', 'Writing Score']

        for ax, col, title, color in zip(axes, subjects, titles, colors):
            sns.histplot(self.df[col], bins=25, kde=True, color=color, ax=ax, alpha=0.7)
            ax.axvline(self.df[col].mean(), color='black', linestyle='--', linewidth=1.5,
                       label=f'Mean: {self.df[col].mean():.1f}')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Score')
            ax.legend()

        fig.suptitle('Distribution of Student Scores', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "score_distributions.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numeric features."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, "correlation_heatmap.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")
        return corr

    def plot_gender_comparison(self):
        """Compare scores across genders."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        subjects = ['math_score', 'reading_score', 'writing_score']
        titles = ['Math Score by Gender', 'Reading Score by Gender', 'Writing Score by Gender']

        for ax, col, title in zip(axes, subjects, titles):
            sns.boxplot(x='gender', y=col, data=self.df, ax=ax, palette='Set2')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Score')

        fig.suptitle('Gender-wise Score Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "gender_comparison.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_parental_education_impact(self):
        """Show impact of parental education on average score."""
        if 'average_score' not in self.df.columns:
            self.df['average_score'] = (
                (self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']) / 3
            ).round(1)

        edu_order = [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ]
        # Filter to only include existing education levels
        existing_levels = [e for e in edu_order if e in self.df['parental_level_of_education'].unique()]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            x='parental_level_of_education', y='average_score',
            data=self.df, order=existing_levels, palette='viridis', ax=ax
        )
        ax.set_title("Impact of Parental Education on Student Performance",
                      fontsize=14, fontweight='bold')
        ax.set_xlabel("Parental Education Level", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=12)
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        path = os.path.join(self.output_dir, "parental_education_impact.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_test_prep_effect(self):
        """Show the effect of test preparation course on scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        melted = self.df.melt(
            id_vars=['test_preparation_course'],
            value_vars=['math_score', 'reading_score', 'writing_score'],
            var_name='Subject', value_name='Score'
        )
        melted['Subject'] = melted['Subject'].str.replace('_score', '').str.title()

        sns.barplot(x='Subject', y='Score', hue='test_preparation_course',
                    data=melted, palette='Set1', ax=ax, ci=95)
        ax.set_title("Effect of Test Preparation on Scores",
                      fontsize=14, fontweight='bold')
        ax.set_ylabel("Average Score")
        ax.legend(title="Test Prep")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "test_prep_effect.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_lunch_impact(self):
        """Show impact of lunch type (socioeconomic proxy) on performance."""
        if 'average_score' not in self.df.columns:
            self.df['average_score'] = (
                (self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']) / 3
            ).round(1)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(x='lunch', y='average_score', data=self.df,
                       palette='coolwarm', ax=ax, inner='box')
        ax.set_title("Lunch Type vs Average Score (Socioeconomic Indicator)",
                      fontsize=14, fontweight='bold')
        ax.set_xlabel("Lunch Type")
        ax.set_ylabel("Average Score")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "lunch_impact.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_study_hours_vs_scores(self):
        """Scatter plot: Study hours vs. average score."""
        if 'average_score' not in self.df.columns:
            self.df['average_score'] = (
                (self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']) / 3
            ).round(1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='study_hours_per_week', y='average_score', data=self.df,
                        hue='test_preparation_course', palette='deep',
                        alpha=0.6, s=50, ax=ax)
        # Add trend line
        valid = self.df.dropna(subset=['study_hours_per_week', 'average_score'])
        z = np.polyfit(valid['study_hours_per_week'], valid['average_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['study_hours_per_week'].min(), valid['study_hours_per_week'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Trend')

        ax.set_title("Study Hours vs Average Score", fontsize=14, fontweight='bold')
        ax.set_xlabel("Study Hours per Week")
        ax.set_ylabel("Average Score")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, "study_hours_vs_scores.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def plot_performance_distribution(self):
        """Pie chart of performance label distribution."""
        if 'average_score' not in self.df.columns:
            self.df['average_score'] = (
                (self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']) / 3
            ).round(1)

        def label_perf(avg):
            if avg >= 90: return 'Excellent'
            elif avg >= 75: return 'Good'
            elif avg >= 60: return 'Average'
            elif avg >= 40: return 'Below Average'
            else: return 'Poor'

        self.df['performance_label'] = self.df['average_score'].apply(label_perf)
        counts = self.df['performance_label'].value_counts()

        colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors[:len(counts)], startangle=140,
            textprops={'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title("Student Performance Distribution",
                      fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, "performance_distribution.png")
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {path}")

    def generate_insights(self):
        """Print key insights discovered from EDA."""
        if 'average_score' not in self.df.columns:
            self.df['average_score'] = (
                (self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']) / 3
            ).round(1)

        print("\n" + "=" * 60)
        print("💡 KEY INSIGHTS FROM EDA")
        print("=" * 60)

        # 1. Gender insights
        gender_avg = self.df.groupby('gender')['average_score'].mean()
        print(f"\n1️⃣  Gender Performance:")
        for g, s in gender_avg.items():
            print(f"   • {g.title()}: {s:.1f} average score")

        # 2. Test prep impact
        prep_avg = self.df.groupby('test_preparation_course')['average_score'].mean()
        prep_diff = prep_avg.get('completed', 0) - prep_avg.get('none', 0)
        print(f"\n2️⃣  Test Preparation Impact:")
        print(f"   • Students who completed prep score {prep_diff:.1f} points higher on average")

        # 3. Lunch/socioeconomic impact
        lunch_avg = self.df.groupby('lunch')['average_score'].mean()
        print(f"\n3️⃣  Socioeconomic Impact (Lunch Type):")
        for l, s in lunch_avg.items():
            print(f"   • {l.title()}: {s:.1f} average score")

        # 4. Top correlated features
        numeric_df = self.df.select_dtypes(include=[np.number])
        if 'average_score' in numeric_df.columns:
            corr_with_avg = numeric_df.corr()['average_score'].drop('average_score').abs().sort_values(ascending=False)
            print(f"\n4️⃣  Top Factors Affecting Performance:")
            for feat, val in corr_with_avg.head(5).items():
                print(f"   • {feat}: correlation = {val:.3f}")

        # 5. Study hours insight
        if 'study_hours_per_week' in self.df.columns:
            valid = self.df.dropna(subset=['study_hours_per_week'])
            high_study = valid[valid['study_hours_per_week'] > 20]['average_score'].mean()
            low_study = valid[valid['study_hours_per_week'] < 10]['average_score'].mean()
            print(f"\n5️⃣  Study Hours Impact:")
            print(f"   • Students studying >20 hrs/week: {high_study:.1f} avg score")
            print(f"   • Students studying <10 hrs/week: {low_study:.1f} avg score")
            print(f"   • Difference: {high_study - low_study:.1f} points")

        print("\n" + "=" * 60)

    def run_full_eda(self):
        """Execute all EDA steps."""
        print("=" * 60)
        print("🔬 RUNNING FULL EXPLORATORY DATA ANALYSIS")
        print("=" * 60 + "\n")

        self.basic_statistics()
        self.plot_score_distributions()
        self.plot_correlation_heatmap()
        self.plot_gender_comparison()
        self.plot_parental_education_impact()
        self.plot_test_prep_effect()
        self.plot_lunch_impact()
        self.plot_study_hours_vs_scores()
        self.plot_performance_distribution()
        self.generate_insights()

        print(f"\n🎉 All charts saved to: {self.output_dir}/")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")
    df = pd.read_csv(data_path)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
    analyzer = EDAAnalyzer(df, output_dir=output_dir)
    analyzer.run_full_eda()
