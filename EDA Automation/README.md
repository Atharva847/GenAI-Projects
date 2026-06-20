# 📊 Automated EDA & Data Preprocessing App

An interactive web application built with **Streamlit** that automates Exploratory Data Analysis (EDA), visualizes relationships, applies machine learning preprocessing steps, and features an **AI Data Scientist Assistant** powered by Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

| Tab | What It Does |
|-----|-------------|
| **📋 Overview** | Dataset properties — rows, columns, data types, index range, memory usage, and unique value analysis |
| **📈 Statistics** | Descriptive stats, interactive missing value summaries, Skewness/Kurtosis plots, and IQR-based outlier detection |
| **🔗 Correlation** | Interactive Pearson correlation heatmap highlighting the strongest numeric associations |
| **📉 Interactive Charts** | Plotly-powered Histograms + KDE, Scatter Plots, Box Plots, Stacked Bar Charts, and Donut Charts |
| **🤖 AI Insights** | Chat with your data using OpenRouter LLMs (Gemini, GPT-4o-mini, DeepSeek) backed by a local FAISS vector store |
| **⚙️ Transform** | Missing value imputation, Label/One-Hot encoding, feature scaling (Standard, MinMax, Robust), and advanced Feature Selection (Variance Threshold, RFE, SelectKBest, Lasso) |
| **💾 Get Code** | Export all transformation steps and charts as a ready-to-run `.py` script or `.ipynb` Jupyter Notebook |

---

## 🛠️ Installation & Setup

> **Requires Python 3.10+**

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**Windows — Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Windows — PowerShell / VS Code Terminal:**
```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the App

Once your virtual environment is active and dependencies are installed:

```bash
streamlit run app.py
```

### Troubleshooting: Global Python Conflict (Windows)

If your terminal resolves to the global Python instead of the venv, launch Streamlit directly:

**Command Prompt:**
```cmd
"C:\Users\<YourName>\Desktop\GenAI Projects\EDA Automation\venv\Scripts\streamlit.exe" run app.py
```

**PowerShell / VS Code Terminal:**
```powershell
& "C:\Users\<YourName>\Desktop\GenAI Projects\EDA Automation\venv\Scripts\streamlit.exe" run app.py
```

---

## 📂 Project Structure

```
EDA Automation/
│
├── app.py                  # Core Streamlit router & sidebar
├── requirements.txt        # All package dependencies
│
└── views/                  # Modular page components
    ├── __init__.py         # Package initialization
    ├── overview.py         # Dataset summary and metadata display
    ├── statistics.py       # Missing values, distributions, IQR checks
    ├── correlation.py      # Heatmaps and correlation matrices
    ├── charts.py           # Plotly-driven dynamic graphs
    ├── data_transform.py   # Cleaning, encoding, scaling, feature selection
    ├── llm_assistant.py    # FAISS vector indexing + OpenRouter LLM chatbot
    └── code_generator.py   # Code exporter (.py and .ipynb formats)
```

---

## 🔑 AI Assistant Setup

To use the **🤖 AI Insights** tab:

1. Obtain an [OpenRouter API Key](https://openrouter.ai/keys).
2. Enter your key in the sidebar when prompted.
3. Select your preferred LLM model from the sidebar dropdown (e.g., Gemini, GPT-4o-mini, DeepSeek).
4. Ask any question — the assistant queries structured dataset summaries indexed in the local **FAISS** vector store via **LangChain RAG**.

---

## 🧰 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Visualizations:** [Plotly](https://plotly.com/python/)
- **ML & Preprocessing:** [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **AI / RAG:** [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [OpenRouter](https://openrouter.ai/)
- **Code Export:** [nbformat](https://nbformat.readthedocs.io/) (Jupyter Notebook generation)

---

## 📌 Notes

- The FAISS index is built **locally at runtime** from your uploaded dataset — no data leaves your machine except for LLM API calls.
- The **Get Code** tab generates clean, reproducible Python code reflecting every transformation you applied in the session.
- Scaling and encoding operations are applied **non-destructively** — your original data is preserved until you explicitly download the transformed version.

---

## 📄 License

This project is for personal / educational use. Add your preferred license here.
