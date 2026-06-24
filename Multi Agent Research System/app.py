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

st.markdown(
    """
    <style>

[data-testid="stChatMessage"] h1,
[data-testid="stExpanderDetails"] h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.8rem !important;
}

[data-testid="stChatMessage"] h2,
[data-testid="stExpanderDetails"] h2 {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.7rem !important;
}

[data-testid="stChatMessage"] h3,
[data-testid="stExpanderDetails"] h3 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] li {
    font-size: 1rem !important;
    line-height: 1.8 !important;
}

[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol,
[data-testid="stExpanderDetails"] ul,
[data-testid="stExpanderDetails"] ol {
    margin-left: 1.5rem !important;
    padding-left: 0.6rem !important;
}

.report-label {
    font-size: 1rem;
    font-weight: 700;
    color: #888;
    margin-bottom: 0.5rem;
}

</style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SESSION STATE
# ==========================================
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox(
        "LLM Provider",
        ["OpenRouter", "Ollama"],
    )
    api_key = ""
    ollama_base_url = "http://localhost:11434/v1"
    if provider == "OpenRouter":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-xxxxxxxx",
        )
    else:
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434/v1",
            placeholder="http://localhost:11434/v1",
        )
        st.caption("Run Ollama locally. Cloud models like `gemma4:31b-cloud` use the same URL.")
    openrouter_models = [
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "openrouter/owl-alpha",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemini-2.5-flash",
        "anthropic/claude-3.5-sonnet",
    ]
    ollama_models = [
        "gemma4:31b-cloud",
        "llama3.3:70b",
        "llama3.2:3b",
        "mistral:7b",
        "qwen2.5:7b",
        "deepseek-r1:8b",
    ]
    model_options = ollama_models if provider == "Ollama" else openrouter_models
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
            placeholder=(
                "e.g. gemma4:31b-cloud"
                if provider == "Ollama"
                else "e.g. qwen/qwen3-235b-a22b"
            ),
        )
    final_model = (
        custom_model.strip()
        if use_custom_model and custom_model.strip()
        else selected_model
    )
    st.divider()
    st.write("### Current Configuration")
    st.write(f"**Provider:** `{provider}`")
    st.write(f"**Model:** `{final_model}`")

    st.divider()

    st.subheader("💬 Chats")

    if st.button("➕ New Chat"):
        chat_name = f"Research #{len(st.session_state.chats)+1}"
        st.session_state.chats[chat_name] = []
        st.session_state.current_chat = chat_name

    for chat_name in st.session_state.chats:
        if st.button(chat_name):
            st.session_state.current_chat = chat_name

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
# DISPLAY HISTORY
# ==========================================
if st.session_state.current_chat is None:
    st.info("👈 Create a New Chat")
    st.stop()

for msg in st.session_state.chats[st.session_state.current_chat]:
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
    if provider == "OpenRouter" and not api_key:
        st.error("Please enter your OpenRouter API Key.")
        st.stop()
    st.session_state.chats[st.session_state.current_chat].append(
        {
            "role": "user",
            "content": query,
        }
    )
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        search_slot = st.empty()
        scrape_slot = st.empty()
        report_slot = st.empty()
        critic_slot = st.empty()

        def on_step(step, state):
            if step == "search":
                with search_slot.container():
                    with st.expander("🔍 Search Agent Results"):
                        st.markdown(state["search_results"])
            elif step == "scrape":
                with scrape_slot.container():
                    with st.expander("📄 Reader Agent (Scraped Content)"):
                        st.markdown(state["scraped_content"])
            elif step == "feedback":
                with critic_slot.container():
                    with st.expander("🧐 Critic Review"):
                        st.markdown(state["feedback"])

        def on_stream(report_text):
            with report_slot.container():
                st.markdown(
                    '<p class="report-label">📝 Research Report</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(report_text)

        try:
            with st.spinner("🔍 Researching..."):
                result = run_research_pipeline(
                    topic=query,
                    model_name=final_model,
                    api_key=api_key or None,
                    provider=provider.lower(),
                    ollama_base_url=ollama_base_url.strip(),
                    on_step=on_step,
                    on_stream=on_stream,
                )
            report = result.get("report", "No report generated.")
            st.session_state.chats[st.session_state.current_chat].append(
                {
                    "role": "assistant",
                    "content": report,
                }
            )
        except Exception as e:
            st.error(f"❌ {e}")
