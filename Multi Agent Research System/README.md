# Multi-Agent Research System 🔬

A powerful, sequential multi-agent research assistant built with LangChain, OpenRouter LLMs, Tavily Search, and Streamlit.

## 🏗️ System Architecture

The pipeline runs sequentially across multiple stages to gather, refine, write, and audit research reports:

```
                  ┌──────────────────────┐
                  │   User Input Topic   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  1. SEARCH AGENT     │ ──► Searches the web via Tavily API
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  2. READER AGENT     │ ──► Selects & scrapes best URL (BS4)
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  3. WRITER CHAIN     │ ──► Generates detailed Markdown report
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  4. CRITIC CHAIN     │ ──► Reviews, rates & gives feedback
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Streamlit UI Output  │
                  └──────────────────────┘
```

## 🚀 Quick Start & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multi-agent-research-system.git
cd multi-agent-research-system
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your Tavily API Key:

```
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

## ⚙️ Configuration & Usage

- **API Credentials:** Enter your OpenRouter or any Custom Model API Key in the sidebar.
- **Choose Model:** Select from the dropdown (e.g., Llama 3.3, GPT-4o-Mini, Claude 3.5 Sonnet) or supply a custom model ID from OpenRouter.
- **Query:** Enter your prompt in the chat box. The pipeline will search, scrape, compose, and critique the topic in real-time.
