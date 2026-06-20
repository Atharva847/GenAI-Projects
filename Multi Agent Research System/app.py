import streamlit as st
from pipeline import run_research_pipeline

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Multi Agent Research System",
    page_icon="🔬",
    layout="wide",
)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-xxxxxxxx",
    )
    model_options = [
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "openrouter/owl-alpha",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemini-2.5-flash",
        "anthropic/claude-3.5-sonnet"
    ]
    selected_model = st.selectbox(
        "Select Model",
        model_options,
    )
    st.divider()
    use_custom_model = st.checkbox(
        "Use Custom Model ID"
    )
    custom_model = ""
    if use_custom_model:
        custom_model = st.text_input(
            "Custom Model ID",
            placeholder="e.g. qwen/qwen3-235b-a22b"
        )
    final_model = (
        custom_model.strip()
        if use_custom_model and custom_model.strip()
        else selected_model
    )
    st.divider()
    st.write("### Current Configuration")
    st.write(f"**Model:** `{final_model}`")

# ==========================================
# TITLE
# ==========================================
st.markdown(
    "<h1 style='text-align:center;'>🔬 Multi Agent Research System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Search • Read • Write • Review</p>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================
# SESSION STATE
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# DISPLAY HISTORY
# ==========================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# INPUT
# ==========================================
query = st.chat_input("Ask a research question...")

# ==========================================
# RUN
# ==========================================
if query:
    if not api_key:
        st.error("Please enter your OpenRouter API Key.")
        st.stop()
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 Researching..."):
                result = run_research_pipeline(
                    topic=query,
                    model_name=final_model,
                    api_key=api_key,
                )
            report = result.get("report", "No report generated.")
            feedback = result.get("feedback", "")
            st.markdown(report)
            if feedback:
                with st.expander("🧐 Critic Review"):
                    st.markdown(feedback)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": report,
                }
            )
        except Exception as e:
            st.error(f"❌ {e}")