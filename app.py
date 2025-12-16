"""
Streamlit Web Interface for Health Data GenAI Analysis
Updated for SQLite-based preprocessing and GenAI pipeline with automatic API key.
"""

import sys
import os
from datetime import datetime

# Add src/ to Python path so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

# ================== IMPORTS ================== #
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from genai_pipeline import HealthDataGenAIPipeline

# ================== ENV VARIABLES ================== #
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment")

# ============================================================================ #
# PAGE CONFIGURATION
# ============================================================================ #
st.set_page_config(
    page_title="Health Data Analysis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================ #
# CUSTOM CSS
# ============================================================================ #
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; font-weight: bold; }
    .result-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; }
    .success-message { color: #28a745; font-weight: bold; }
    .error-message { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================ #
# INITIALIZE SESSION STATE
# ============================================================================ #
if 'pipeline' not in st.session_state:
    with st.spinner("🚀 Initializing AI pipeline..."):
        st.session_state.pipeline = HealthDataGenAIPipeline(api_key=API_KEY)

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# ============================================================================ #
# SIDEBAR - QUERY HISTORY
# ============================================================================ #
st.sidebar.header("📜 Query History")
if st.session_state.query_history:
    for i, query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
        with st.sidebar.expander(f"Query {len(st.session_state.query_history) - i + 1}"):
            st.write(query['question'])
            st.caption(query['timestamp'])
else:
    st.sidebar.info("No queries yet")

# ============================================================================ #
# MAIN HEADER
# ============================================================================ #
st.markdown('<h1 class="main-header">🏥 Health Data Analysis AI Assistant</h1>', unsafe_allow_html=True)

# Example questions
examples = [
    "What is the average BMI of male patients?",
    "Do smokers have lower physical activity than non-smokers?",
    "What percentage of patients have abnormal blood pressure?",
    "Show correlation between stress levels and blood pressure",
    "What is the average age of patients with chronic kidney disease?",
    "Compare health outcomes between highly active and sedentary people",
    "What lifestyle factors are associated with high blood pressure?",
    "Show me patients with BMI over 30 and low physical activity"
]

# ============================================================================ #
# QUERY INTERFACE
# ============================================================================ #
st.subheader("💬 Ask Your Question")
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the average BMI of smokers?",
        label_visibility="collapsed"
    )

with col2:
    selected_example = st.selectbox(
        "Or choose an example:",
        ["Select an example..."] + examples,
        label_visibility="collapsed"
    )
    if selected_example != "Select an example...":
        user_question = selected_example

analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ============================================================================ #
# PROCESS QUERY
# ============================================================================ #
if analyze_button and user_question:
    if not user_question.strip():
        st.error("Please enter a question!")
    else:
        col_query, col_response = st.columns([1, 2])
        with st.spinner("🤔 Analyzing your question..."):
            try:
                result = st.session_state.pipeline.process_query(user_question, verbose=False)

                if result['success']:
                    # Save to history
                    st.session_state.query_history.append({
                        'question': user_question,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'response': result['response']
                    })

                    # Show SQL and results
                    with col_query:
                        st.subheader("🔧 Generated SQL Query")
                        st.code(result['sql_query'], language='sql')

                        if isinstance(result['results'], pd.DataFrame) and not result['results'].empty:
                            with st.expander("📊 View Raw Data"):
                                st.dataframe(result['results'], use_container_width=True)
                                csv = result['results'].to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                    # Show AI response
                    with col_response:
                        st.subheader("💡 AI Response")
                        st.markdown(result['response'])
                        st.success("✅ Analysis completed successfully!")
                        with st.expander("📏 Evaluation Scores"):
                            st.write(result.get("evaluation", {}))
                else:
                    st.error(f"❌ Error: {result['response']}")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# ============================================================================ #
# ADDITIONAL TABS
# ============================================================================ #
st.divider()
tab1, tab2 = st.tabs(["📈 Quick Stats", "ℹ️ About"])

# ================= Quick Stats Tab ================= #
with tab1:
    st.subheader("Quick Dataset Statistics")
    df1 = st.session_state.pipeline.df1
    df_stats = st.session_state.pipeline.df_stats

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", df1.shape[0])
        st.metric("Average Age", f"{df1['Age'].mean():.1f} years")
    with col2:
        st.metric("Male Patients", f"{(df1['Sex'] == 0).sum()}")
        st.metric("Female Patients", f"{(df1['Sex'] == 1).sum()}")
    with col3:
        st.metric("Smokers", f"{(df1['Smoking'] == 1).sum()}")
        st.metric("BP Abnormal", f"{(df1['Blood_Pressure_Abnormality'] == 1).sum()}")
    with col4:
        st.metric("Avg BMI", f"{df1['BMI'].mean():.1f}")
        st.metric("Avg Steps/Day", f"{df_stats['Avg_Physical_Activity'].mean():.0f}")

# ================= About Tab ================= #
with tab2:
    st.subheader("About This Application")
    st.markdown("""
    ### 🏥 Health Data Analysis AI Assistant
    This application uses advanced AI (Google Gemini) to analyze health data through natural language queries.

    **Features:**
    - 🤖 Natural language query processing
    - 📊 Automatic SQL query generation
    - 💡 Contextual health insights
    - 📈 Data visualization
    - 📥 Export results

    **Technology Stack:**
    - Google Gemini AI for NLP
    - Python (Pandas, SQLite) for data processing
    - Streamlit for web interface
    - SQL for data querying

    **Privacy Notice:** All data processing happens locally. Your queries and data are not stored permanently.
    """)

# ============================================================================ #
# FOOTER
# ============================================================================ #
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Health Data Analysis AI Assistant</p>
    <p style='font-size: 0.8rem;'>Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
