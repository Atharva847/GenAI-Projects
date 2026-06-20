import os
import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


MODEL_OPTIONS = [
    "openrouter/owl-alpha",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
]

SYSTEM_PROMPT = """You are an elite, battle-tested Principal Data Scientist and Lead Statistical Analyst.

Your mission is to perform deep, rigorous diagnostic and prescriptive analytics on the provided dataset.

CRITICAL INSTRUCTIONS:
1. Grounded Accuracy: Base your factual claims and metrics strictly on the dataset context provided below. Never invent numbers.
2. If the context is completely silent on a metric or column, declare: "I cannot find specific evidence of this in the dataset."
3. Domain Integration: While you must remain grounded in the dataset's numbers, you SHOULD apply master-level data science domain expertise to propose theoretical hypotheses, explain mathematical patterns (like log-transformations for high skewness), or suggest specific ML modeling pipelines. Always clearly separate factual context from analytical hypotheses.

When analyzing, systematically look for:
- 📈 Multi-Variable Interactions: How categorical dimensions might segment numerical distributions (e.g., Simpson's Paradox or group imbalances).
- 🧬 Distribution Dynamics: Identify heavily skewed features (|Skewness| > 1) and recommend mathematical corrections (Log, Box-Cox, Yeo-Johnson).
- 🔗 Multicollinearity & Correlation Profiles: Highlight strong bivariate associations (|r| >= 0.70) and potential redundant features.
- ⚠️ Data Anomalies & Friction Points: Diagnose extreme outliers (via IQR boundaries), data-quality leaks, high missingness ratios, or high-cardinality categories.
- 🔧 Preprocessing & Engineering Blueprints: Detail precisely which encoding (Label vs. One-Hot), scaling (Standard, MinMax, Robust), or selection strategies (RFE, Lasso) are mathematically suited to this dataset structure.
- 🤖 Predictive Architectures: Formulate comprehensive ML strategies (regression, classification, clustering, or time-series) with concrete cross-validation and evaluation metrics tailored to the target.

Formatting Guidelines:
- Use clean, structured Markdown with short, impactful headers and clean bullet points.
- Bold important **columns**, **metrics**, **findings**, and **algorithms**.
- Incorporate formulas in LaTeX format (e.g., $IQR = Q_3 - Q_1$) when explaining statistical definitions or transformations.
- Ensure the tone remains exceptionally professional, advisory, and mathematically precise."""


def _dataset_text(df):
    num_df = df.select_dtypes(include="number")
    extra = ""
    if not num_df.empty:
        extra = f"""
Skewness:
{num_df.skew()}

Correlation:
{num_df.corr().round(3)}
"""
    return f"""
Shape: {df.shape}

Columns: {list(df.columns)}

Data Types:
{df.dtypes}

Missing Values:
{df.isnull().sum()}

Duplicate Rows: {df.duplicated().sum()}

Statistics:
{df.describe(include='all')}
{extra}
First 10 Rows:
{df.head(10)}
"""


def _cache_key(df, model):
    return f"{len(df)}_{list(df.columns)}_{model}"


def build_rag(df, api_key, model):
    key = _cache_key(df, model)
    if st.session_state.get("rag_cache_key") == key and "rag_chain" in st.session_state:
        return st.session_state.rag_chain

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts([_dataset_text(df)], embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        streaming=True,
    )

    chain = {"retriever": retriever, "llm": llm}
    st.session_state.rag_chain = chain
    st.session_state.rag_cache_key = key
    return chain


def stream_answer(chain, question):
    docs = chain["retriever"].invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""{SYSTEM_PROMPT}

Dataset context:
{context}

Question:
{question}
"""
    for chunk in chain["llm"].stream(prompt):
        if chunk.content:
            yield chunk.content


def render_sidebar():
    st.markdown("### 🤖 AI Settings")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        key="rag_api_key",
    )
    model = st.selectbox("Model", MODEL_OPTIONS, key="rag_model")
    if st.button("Clear chat", use_container_width=True, key="clear_chat"):
        st.session_state.chat_messages = []
        st.session_state.pop("rag_chain", None)
        st.session_state.pop("rag_cache_key", None)
        st.rerun()
    return api_key, model


def render(df, api_key, model):
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your dataset...")
    if not question:
        return

    if not api_key:
        st.error("Enter your OpenRouter API key in the sidebar.")
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chain = build_rag(df, api_key, model)
            answer = st.write_stream(stream_answer(chain, question))
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")
