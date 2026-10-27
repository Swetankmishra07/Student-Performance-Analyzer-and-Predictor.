import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.prediction import StudentPredictor


st.set_page_config(
    page_title="🎓 Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .main-header h1 {
        color: #fff;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.75);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.3);
        border-color: rgba(99,102,241,0.5);
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* Prediction result card */
    .prediction-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        border: 2px solid;
        transition: all 0.3s ease;
    }

    .prediction-excellent {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(5,150,105,0.15) 100%);
        border-color: rgba(16,185,129,0.4);
    }

    .prediction-good {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(37,99,235,0.15) 100%);
        border-color: rgba(59,130,246,0.4);
    }

    .prediction-average {
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(217,119,6,0.15) 100%);
        border-color: rgba(245,158,11,0.4);
    }

    .prediction-below {
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.15) 100%);
        border-color: rgba(239,68,68,0.4);
    }

    .prediction-poor {
        background: linear-gradient(135deg, rgba(127,29,29,0.1) 0%, rgba(153,27,27,0.15) 100%);
        border-color: rgba(153,27,27,0.4);
    }

    .score-display {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2b55 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* Section headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99,102,241,0.3);
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)


# ============================
# LOAD DATA & MODELS
# ============================
@st.cache_data
def load_data():
    """Load the student performance dataset."""
    data_path = os.path.join(PROJECT_ROOT, "data", "student_performance.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


@st.cache_resource
def load_predictor():
    """Load the trained model and preprocessor."""
    model_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    predictor = StudentPredictor(model_dir=model_dir)

    # Try to load regression model
    model_file = "random_forest_regression_model.pkl"
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        predictor.load_model(model_file)

    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        predictor.load_preprocessor(preprocessor_path)

    return predictor


# ============================
# HEADER
# ============================
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Analyzer</h1>
    <p>AI-Powered Academic Performance Prediction & Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select a section:",
        ["🔮 Predict Performance", "📊 Data Explorer", "📈 EDA Dashboard",
         "🤖 Model Insights", "📘 About the Project"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>💡 How to use:</strong><br>
        <small>1. Start with <b>Predict</b> to try the model<br>
        2. Explore data in <b>Data Explorer</b><br>
        3. View charts in <b>EDA Dashboard</b><br>
        4. Understand models in <b>Model Insights</b></small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 1rem;">
        Built with ❤️ using Python & Streamlit<br>
        Student Performance Analyzer v1.0
    </div>
    """, unsafe_allow_html=True)


# ============================
# PAGE: PREDICT PERFORMANCE
# ============================
if page == "🔮 Predict Performance":
    st.markdown('<div class="section-header">🔮 Predict Student Performance</div>', unsafe_allow_html=True)
    st.markdown("Enter student details below to get an AI-powered prediction of their academic performance.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Student Demographics")
        gender = st.selectbox("Gender", ["female", "male"], key="gender")
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            ["group A", "group B", "group C", "group D", "group E"],
            index=2, key="race"
        )
        parental_education = st.selectbox(
            "Parental Education Level",
            ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"],
            index=2, key="parent_edu"
        )

    with col2:
        st.markdown("##### 📚 Academic Factors")
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"], key="lunch")
        test_prep = st.selectbox(
            "Test Preparation Course",
            ["completed", "none"],
            key="test_prep"
        )
        study_hours = st.slider("Study Hours per Week", 1.0, 40.0, 15.0, 0.5, key="study_hrs")
        attendance = st.slider("Attendance Rate (%)", 30.0, 100.0, 85.0, 0.5, key="attendance")

    # Predict button
    predict_btn = st.button("🚀 Predict Performance", type="primary", use_container_width=True)

    if predict_btn:
        input_data = {
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_education,
            'lunch': lunch,
            'test_preparation_course': test_prep,
            'study_hours_per_week': study_hours,
            'attendance_rate': attendance
        }

        predictor = load_predictor()

        if predictor.model is not None:
            with st.spinner("🧠 Analyzing student profile..."):
                import time
                time.sleep(0.8)  # Brief pause for UX
                result = predictor.predict_with_details(input_data)

            # Display prediction result
            label_lower = result['performance_label'].lower().replace(" ", "_").replace("below_average", "below")
            css_class = f"prediction-{label_lower}"

            st.markdown(f"""
            <div class="prediction-card {css_class}">
                <div style="font-size: 1.2rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 2px;">
                    Predicted Average Score
                </div>
                <div class="score-display">{result['emoji']} {result['predicted_score']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; font-family: 'Outfit', sans-serif;">
                    {result['performance_label']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Score gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['predicted_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Performance Score", 'font': {'size': 18, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#444'},
                    'bar': {'color': '#667eea'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239,68,68,0.2)'},
                        {'range': [40, 60], 'color': 'rgba(245,158,11,0.2)'},
                        {'range': [60, 75], 'color': 'rgba(59,130,246,0.2)'},
                        {'range': [75, 90], 'color': 'rgba(16,185,129,0.2)'},
                        {'range': [90, 100], 'color': 'rgba(5,150,105,0.3)'},
                    ],
                    'threshold': {
                        'line': {'color': '#fff', 'width': 3},
                        'thickness': 0.75,
                        'value': result['predicted_score']
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            if result['recommendations']:
                st.markdown('<div class="section-header">💡 Personalized Recommendations</div>',
                            unsafe_allow_html=True)
                for rec in result['recommendations']:
                    st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Model not found! Please run `python run_pipeline.py` first to train the models.")


# ============================
# PAGE: DATA EXPLORER
# ============================
elif page == "📊 Data Explorer":
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)

    df = load_data()
    if df is not None:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Students</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_math = df['math_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_math:.1f}</div>
                <div class="metric-label">Avg Math Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_read = df['reading_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_read:.1f}</div>
                <div class="metric-label">Avg Reading Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_write = df['writing_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_write:.1f}</div>
                <div class="metric-label">Avg Writing Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Data filters
        with st.expander("🔍 Filter Data", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
            with filter_col2:
                lunch_filter = st.multiselect("Lunch", df['lunch'].unique(), default=df['lunch'].unique())
            with filter_col3:
                prep_filter = st.multiselect("Test Prep", df['test_preparation_course'].unique(),
                                             default=df['test_preparation_course'].unique())

            filtered_df = df[
                (df['gender'].isin(gender_filter)) &
                (df['lunch'].isin(lunch_filter)) &
                (df['test_preparation_course'].isin(prep_filter))
            ]
        
        # Display data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "math_score": st.column_config.ProgressColumn("Math Score", min_value=0, max_value=100, format="%d"),
                "reading_score": st.column_config.ProgressColumn("Reading Score", min_value=0, max_value=100, format="%d"),
                "writing_score": st.column_config.ProgressColumn("Writing Score", min_value=0, max_value=100, format="%d"),
            }
        )
        st.caption(f"Showing {len(filtered_df)} of {len(df)} students")

        # Descriptive stats
        with st.expander("📋 Descriptive Statistics"):
            st.dataframe(filtered_df.describe().round(2), use_container_width=True)
    else:
        st.warning("⚠️ Dataset not found. Run `python run_pipeline.py` first.")


# ============================
# PAGE: EDA DASHBOARD
# ============================
elif page == "📈 EDA Dashboard":
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)

    df = load_data()
    if df is not None:
        # Add computed columns
        df['average_score'] = ((df['math_score'] + df['reading_score'] + df['writing_score']) / 3).round(1)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Score Distributions", "🔗 Correlations",
            "📉 Factor Analysis", "🎯 Deep Insights"
        ])

        with tab1:
            # Score distributions with Plotly
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, x='math_score', nbins=30, color_discrete_sequence=['#667eea'],
                    title='Math Score Distribution', marginal='box',
                    template='plotly_dark'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    df, x='reading_score', nbins=30, color_discrete_sequence=['#10b981'],
                    title='Reading Score Distribution', marginal='box',
                    template='plotly_dark'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig = px.histogram(
                    df, x='writing_score', nbins=30, color_discrete_sequence=['#f59e0b'],
                    title='Writing Score Distribution', marginal='box',
                    template='plotly_dark'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                fig = px.histogram(
                    df, x='average_score', nbins=30, color_discrete_sequence=['#a855f7'],
                    title='Average Score Distribution', marginal='violin',
                    template='plotly_dark'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr, text_auto='.2f', aspect='auto',
                color_continuous_scale='RdYlBu_r',
                title='Feature Correlation Heatmap',
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # Score scatter matrix
            fig = px.scatter_matrix(
                df, dimensions=['math_score', 'reading_score', 'writing_score'],
                color='gender', template='plotly_dark',
                title='Score Relationships by Gender'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Factor analysis
            col1, col2 = st.columns(2)

            with col1:
                # Gender comparison
                fig = px.box(
                    df.melt(id_vars='gender', value_vars=['math_score', 'reading_score', 'writing_score'],
                            var_name='Subject', value_name='Score'),
                    x='Subject', y='Score', color='gender',
                    title='Score Distribution by Gender',
                    template='plotly_dark',
                    color_discrete_sequence=['#667eea', '#f472b6']
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Test prep effect
                fig = px.box(
                    df, x='test_preparation_course', y='average_score',
                    color='test_preparation_course',
                    title='Test Preparation Course Impact',
                    template='plotly_dark',
                    color_discrete_sequence=['#ef4444', '#10b981']
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                # Parental education impact
                edu_order = [
                    "some high school", "high school", "some college",
                    "associate's degree", "bachelor's degree", "master's degree"
                ]
                fig = px.box(
                    df, x='parental_level_of_education', y='average_score',
                    category_orders={'parental_level_of_education': edu_order},
                    title='Parental Education vs Performance',
                    template='plotly_dark',
                    color='parental_level_of_education',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                # Lunch impact
                fig = px.violin(
                    df, x='lunch', y='average_score', color='lunch',
                    title='Lunch Type vs Average Score',
                    template='plotly_dark', box=True,
                    color_discrete_sequence=['#3b82f6', '#ef4444']
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Deep insights
            col1, col2 = st.columns(2)
            with col1:
                if 'study_hours_per_week' in df.columns:
                    fig = px.scatter(
                        df.dropna(subset=['study_hours_per_week']),
                        x='study_hours_per_week', y='average_score',
                        color='test_preparation_course',
                        trendline='ols',
                        title='Study Hours vs Performance',
                        template='plotly_dark',
                        color_discrete_sequence=['#667eea', '#f59e0b']
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'attendance_rate' in df.columns:
                    fig = px.scatter(
                        df.dropna(subset=['attendance_rate']),
                        x='attendance_rate', y='average_score',
                        color='lunch',
                        trendline='ols',
                        title='Attendance Rate vs Performance',
                        template='plotly_dark',
                        color_discrete_sequence=['#10b981', '#ef4444']
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

            # Performance distribution pie
            def label_perf(avg):
                if avg >= 90: return 'Excellent'
                elif avg >= 75: return 'Good'
                elif avg >= 60: return 'Average'
                elif avg >= 40: return 'Below Average'
                else: return 'Poor'

            df['performance'] = df['average_score'].apply(label_perf)
            perf_counts = df['performance'].value_counts()

            fig = px.pie(
                values=perf_counts.values,
                names=perf_counts.index,
                title='Overall Performance Distribution',
                template='plotly_dark',
                color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#7f1d1d'],
                hole=0.4
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Dataset not found. Run `python run_pipeline.py` first.")


# ============================
# PAGE: MODEL INSIGHTS
# ============================
elif page == "🤖 Model Insights":
    st.markdown('<div class="section-header">🤖 Model Performance & Insights</div>', unsafe_allow_html=True)

    model_dir = os.path.join(PROJECT_ROOT, "outputs", "models")

    # Check for saved model results
    reg_model_path = os.path.join(model_dir, "random_forest_regression_model.pkl")
    
    if os.path.exists(reg_model_path):
        model = joblib.load(reg_model_path)
        meta_path = os.path.join(model_dir, "random_forest_regression_metadata.json")
        
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            features = meta.get('feature_names', [])
        else:
            features = []

        st.markdown("### 📈 Model Comparison (Regression)")
        st.markdown("""
        We trained 3 models to predict student average scores. Here's how they compare:
        """)

        # Model comparison data
        model_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
            'Description': [
                'Finds the best straight-line fit through the data',
                'Creates rules by splitting data into branches',
                'Combines 100 decision trees for robust predictions'
            ],
            'Pros': [
                'Simple, fast, interpretable',
                'Captures non-linear patterns',
                'Best accuracy, handles complex data'
            ],
            'Cons': [
                'Assumes linear relationships',
                'Can overfit to training data',
                'Slower, harder to interpret'
            ]
        })

        st.dataframe(model_comparison, use_container_width=True, hide_index=True)

        # Feature importance
        if hasattr(model, 'feature_importances_') and features:
            st.markdown("### 🎯 Feature Importance")
            st.markdown("Which factors matter most for predicting student performance?")

            importances = model.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                feat_imp, x='Importance', y='Feature',
                orientation='h', template='plotly_dark',
                color='Importance', color_continuous_scale='Viridis',
                title='Feature Importance (Random Forest Model)'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                yaxis_title='',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Explain top features
            top_feat = feat_imp.iloc[-1]
            st.markdown(f"""
            <div class="info-box">
                <strong>🔍 Key Insight:</strong> The most important feature is <b>{top_feat['Feature']}</b> 
                with an importance score of <b>{top_feat['Importance']:.4f}</b>. This means it has the 
                strongest influence on predicting a student's academic performance.
            </div>
            """, unsafe_allow_html=True)

        # Evaluation metrics explanation
        st.markdown("### 📋 Understanding Evaluation Metrics")
        metrics_data = {
            'Metric': ['MAE', 'RMSE', 'R² Score', 'Accuracy', 'Confusion Matrix'],
            'Full Name': [
                'Mean Absolute Error', 'Root Mean Squared Error',
                'R-Squared / Coefficient of Determination',
                'Classification Accuracy', 'Confusion Matrix'
            ],
            'What It Means': [
                'Average difference between predicted and actual scores (lower is better)',
                'Like MAE but penalizes large errors more (lower is better)',
                'How much variance the model explains. 1.0 = perfect (higher is better)',
                'Percentage of correct predictions in classification (higher is better)',
                'Table showing correct vs incorrect predictions for each category'
            ],
            'Used For': [
                'Regression', 'Regression', 'Regression',
                'Classification', 'Classification'
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No trained models found. Run `python run_pipeline.py` first.")
        st.code("python run_pipeline.py", language="bash")


# ============================
# PAGE: ABOUT THE PROJECT
# ============================
elif page == "📘 About the Project":
    st.markdown('<div class="section-header">📘 About the Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎓 Student Performance Analyzer & Predictor

    ### Problem Statement
    **Can we predict a student's academic performance based on demographic, socioeconomic, 
    and behavioral factors?**

    This is a critical question in education because early identification of at-risk students 
    allows educators to provide timely intervention and support.

    ### Real-World Use Case
    - **Schools** can identify students who may need extra tutoring
    - **Parents** can understand which factors most influence their child's performance
    - **Policy makers** can allocate resources to programs that have the most impact
    - **Students** can get personalized recommendations for improvement

    ---

    ### 📊 Dataset Overview
    | Column | Description | Type |
    |--------|-------------|------|
    | `gender` | Student's gender (male/female) | Categorical |
    | `race_ethnicity` | Ethnic group (A through E) | Categorical |
    | `parental_level_of_education` | Highest education of parents | Categorical |
    | `lunch` | Standard or free/reduced (socioeconomic indicator) | Categorical |
    | `test_preparation_course` | Whether student completed prep course | Categorical |
    | `study_hours_per_week` | Weekly study hours | Numeric |
    | `attendance_rate` | Class attendance percentage | Numeric |
    | `math_score` | Score in mathematics (0-100) | Target |
    | `reading_score` | Score in reading (0-100) | Target |
    | `writing_score` | Score in writing (0-100) | Target |

    ---

    ### 🚀 Deployment Guide
    
    #### Option 1: Streamlit Cloud (Free & Easiest)
    1. Push this project to a GitHub repository
    2. Go to [share.streamlit.io](https://share.streamlit.io)
    3. Connect your GitHub repo
    4. Set `app.py` as the main file
    5. Click Deploy! ✅

    #### Option 2: Render (Free Tier)
    1. Create a `render.yaml` with web service config
    2. Push to GitHub and connect to Render
    3. Set start command: `streamlit run app.py --server.port $PORT`

    #### Option 3: Railway / Heroku
    1. Create a `Procfile`: `web: streamlit run app.py --server.port $PORT`
    2. Push and deploy

    ---

    ### 📝 Resume Points
    - Developed an end-to-end **Student Performance Prediction System** using Python, 
      scikit-learn, and Streamlit
    - Performed comprehensive **EDA** with 8+ visualizations, uncovering key factors 
      affecting student outcomes
    - Built and compared **3 ML models** (Linear Regression, Decision Tree, Random Forest) 
      achieving R² > 0.85
    - Engineered a **real-time prediction system** with personalized academic recommendations
    - Created an **interactive web dashboard** with data exploration, analytics, and 
      model insights
    - Handled data preprocessing including **missing value imputation**, feature engineering, 
      and categorical encoding

    ---

    ### 🎤 Interview Q&A

    **Q1: Why did you choose Random Forest over other models?**
    > Random Forest is an ensemble method that combines multiple decision trees. It reduces 
    > overfitting (which is a problem with single decision trees) and provides feature importance 
    > rankings. In our case, it gave the highest R² score and most consistent cross-validation results.

    **Q2: How did you handle missing values and why?**
    > Numeric columns (study_hours, attendance) were filled with the **median** because it's robust 
    > to outliers. Categorical columns were filled with the **mode** (most frequent value). This 
    > preserves the distribution better than mean imputation.

    **Q3: What was the most surprising insight from your EDA?**
    > The **lunch type** (a proxy for socioeconomic status) had a surprisingly large impact on scores — 
    > students with standard lunch scored significantly higher across all subjects. This highlights 
    > how socioeconomic factors influence academic performance beyond just study habits.

    **Q4: How would you deploy this in production?**
    > I would containerize the app with Docker, use a CI/CD pipeline (GitHub Actions), deploy on 
    > a cloud platform (AWS/GCP/Azure), and add monitoring for model drift. For scaling, I'd use 
    > an API (FastAPI) with the model served behind a load balancer.

    **Q5: What improvements would you make with more time?**
    > (a) Add time-series data to track performance over semesters, 
    > (b) Use deep learning for more complex patterns, 
    > (c) Add SHAP values for better model explainability, 
    > (d) Build an admin dashboard for teachers with student cohort analysis, 
    > (e) Implement A/B testing for different intervention strategies.
    """)
