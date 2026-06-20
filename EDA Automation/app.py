import streamlit as st
import pandas as pd
import warnings

from views import overview, statistics, correlation, charts, data_transform, llm_assistant, code_generator

warnings.filterwarnings("ignore", category=UserWarning)

PAGES = [
    "📌 Overview",
    "📊 Statistics",
    "📈 Correlation",
    "📉 Interactive Charts",
    "🤖 AI Insights",
    "🔧 Transform Your Data",
    "📥 Get Code",
]

st.set_page_config(page_title="Automated EDA App", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.title("📊 EDA App")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file from the sidebar to get started.")
else:
    df = pd.read_csv(uploaded_file)

    with st.sidebar:
        st.divider()
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")

        api_key, model = None, None
        if page == "🤖 AI Insights":
            st.divider()
            api_key, model = llm_assistant.render_sidebar()

    if page == "📌 Overview":
        overview.render(df)
    elif page == "📊 Statistics":
        statistics.render(df)
    elif page == "📈 Correlation":
        correlation.render(df)
    elif page == "📉 Interactive Charts":
        charts.render(df)
    elif page == "🤖 AI Insights":
        llm_assistant.render(df, api_key, model)
    elif page == "🔧 Transform Your Data":
        data_transform.render(df)
    elif page == "📥 Get Code":
        code_generator.render(uploaded_file.name)
