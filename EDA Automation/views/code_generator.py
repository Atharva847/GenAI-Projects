import json
import streamlit as st


def _transform_code(history):
    lines = []
    for action in history:
        t = action["type"]
        if t == "drop_columns":
            lines.append(f"df = df.drop(columns={action['columns']!r})")
        elif t == "drop_duplicates":
            lines.append("df = df.drop_duplicates()")
        elif t == "missing_values":
            lines.append("\n# Handle missing values")
            for col, strategy in action["strategies"].items():
                if strategy == "Mean":
                    lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
                elif strategy == "Median":
                    lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
                elif strategy == "Mode":
                    lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
        elif t == "label_encoding":
            lines.append("\nfrom sklearn.preprocessing import LabelEncoder")
            lines.append("le = LabelEncoder()")
            for col in action["columns"]:
                if action.get("add_new"):
                    lines.append(f"df['{col}_Encoded'] = le.fit_transform(df['{col}'].astype(str))")
                else:
                    lines.append(f"df['{col}'] = le.fit_transform(df['{col}'].astype(str))")
        elif t == "custom_encoding":
            label_cols = action.get("label_columns", [])
            ohe_cols = action.get("ohe_columns", [])
            le_add_new = action.get("label_add_new", False)
            if label_cols:
                lines.append("\nfrom sklearn.preprocessing import LabelEncoder")
                for col in label_cols:
                    lines.append("le = LabelEncoder()")
                    if le_add_new:
                        lines.append(f"df['{col}_Encoded'] = le.fit_transform(df['{col}'].astype(str))")
                    else:
                        lines.append(f"df['{col}'] = le.fit_transform(df['{col}'].astype(str))")
            if ohe_cols:
                drop_first = action.get("ohe_drop_first", False)
                drop_param = "'first'" if drop_first else "None"
                lines.append("\nfrom sklearn.preprocessing import OneHotEncoder")
                lines.append(f"ohe = OneHotEncoder(sparse_output=False, drop={drop_param}, handle_unknown='ignore')")
                lines.append(f"encoded = ohe.fit_transform(df[{ohe_cols!r}].astype(str))")
                lines.append(
                    "encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out("
                    + repr(ohe_cols) + "), index=df.index)"
                )
                if action.get("ohe_drop_original"):
                    lines.append(f"df = df.drop(columns={ohe_cols!r})")
                lines.append("df = pd.concat([df, encoded_df], axis=1)")
        elif t == "one_hot_encoding":
            cols = action["columns"]
            lines.append("\nfrom sklearn.preprocessing import OneHotEncoder")
            lines.append("ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')")
            lines.append(f"encoded = ohe.fit_transform(df[{cols!r}].astype(str))")
            lines.append("encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(" + repr(cols) + "), index=df.index)")
            if action.get("drop_original"):
                lines.append(f"df = df.drop(columns={cols!r})")
            lines.append("df = pd.concat([df, encoded_df], axis=1)")
        elif t == "scaling":
            method = action["method"]
            cols = action["columns"]
            if method == "Standardization":
                lines.append("\nfrom sklearn.preprocessing import StandardScaler")
                lines.append("scaler = StandardScaler()")
            elif method == "MinMaxScaling":
                lines.append("\nfrom sklearn.preprocessing import MinMaxScaler")
                lines.append("scaler = MinMaxScaler()")
            elif method == "Robust Scaling":
                lines.append("\nfrom sklearn.preprocessing import RobustScaler")
                lines.append("scaler = RobustScaler()")
            lines.append(f"df[{cols!r}] = scaler.fit_transform(df[{cols!r}])")
        elif t == "feature_selection":
            selected = action["selected_features"]
            target = action["target"]
            keep = selected + [target]
            lines.append(f"\n# Feature selection: {action['category']} — {action['method']}")
            lines.append(f"df = df[{keep!r}]")
    return "\n".join(lines)


def _generate_python(filename, history):
    transform_block = _transform_code(history)
    return f'''import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# Load data
df = pd.read_csv("{filename}")

# --- Overview ---
print("Rows:", len(df), "Columns:", len(df.columns))
print(df.head(10))
print(df.dtypes)
print("Unique counts:\\n", df.nunique())
print("Duplicates:", df.duplicated().sum())

# --- Statistics ---
print(df.describe().transpose())
print("Missing values:\\n", df.isnull().sum())

num_df = df.select_dtypes(include=["int", "float"])
if not num_df.empty:
    print("Skewness:\\n", num_df.skew())
    print("Kurtosis:\\n", num_df.kurtosis())

# --- Correlation ---
numeric_df = df.select_dtypes(include="number")
if not numeric_df.empty:
    corr = numeric_df.corr().round(2)
    print("Correlation matrix:\\n", corr)

# --- Charts (example) ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if numeric_cols:
    col = numeric_cols[0]
    fig = px.histogram(df, x=col, title=f"Histogram of {{col}}")
    fig.show()

# --- Transform ---
{transform_block if transform_block else "# No transformations applied"}

print(df.head(10))
'''


def _generate_notebook(filename, history):
    code = _generate_python(filename, history)
    cells = [{"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in code.split("\n")],
              "outputs": [], "execution_count": None}]
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


def render(filename):
    st.subheader("📥 Get Code")
    st.caption("Download Python or Jupyter notebook with EDA + transform steps (no AI/RAG code).")

    history = st.session_state.get("transform_history", [])
    if history:
        st.success(f"✅ {len(history)} transformation step(s) will be included.")
    else:
        st.info("No transformations applied yet. Code will include overview, statistics, correlation, and charts only.")

    py_code = _generate_python(filename, history)
    st.code(py_code, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📄 Download .py", data=py_code, file_name="eda_pipeline.py", mime="text/plain")
    with col2:
        nb_json = json.dumps(_generate_notebook(filename, history), indent=2)
        st.download_button("📓 Download .ipynb", data=nb_json, file_name="eda_pipeline.ipynb", mime="application/json")
