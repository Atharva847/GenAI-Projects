import streamlit as st
import pandas as pd
import numpy as np


def render(df):
    st.markdown("### 📌 Dataset Summary")
    st.markdown("Here's a quick overview of your dataset:")

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            num_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeric Columns", num_cols)
        with col4:
            cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
            st.metric("Categorical Columns", cat_cols)

    st.markdown("---")
    st.markdown("### 🔍 First Look at Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    st.markdown("### 🧾 Column Info Summary")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Dtype": df.dtypes.astype(str).values
    })
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(info_df, use_container_width=True)
    with c2:
        dtype_counts = df.dtypes.value_counts()
        dtype_str = ', '.join([f"{dtype}({count})" for dtype, count in dtype_counts.items()])
        st.markdown(f"*Data Types:* {dtype_str}")
        st.markdown(f"*Index Range:* {df.index.start} to {df.index.stop - 1}")
        mem_kb = round(df.memory_usage().sum() / 1024, 2)
        st.markdown(f"*Memory Usage:* {mem_kb} KB")

    st.markdown("---")
    st.markdown("### 🔢 Unique Values Summary")
    unique_count_df = df.nunique().reset_index()
    unique_count_df.columns = ["Column", "Unique Value Count"]
    st.dataframe(unique_count_df, use_container_width=True)

    with st.expander("🔍 Click to View All Unique Values Per Column"):
        for col in df.columns:
            st.markdown(f"*{col}* — {df[col].nunique()} unique values")
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 25:
                st.write(unique_vals.tolist())
            else:
                st.caption("Too many values to show. Showing first 25:")
                st.write(unique_vals[:25].tolist())

    st.markdown("---")
    st.markdown("### 🔁 Duplicate Rows")
    st.metric("Total Duplicate Rows", df.duplicated().sum())
    st.markdown("---")
    st.markdown("### 🔢 Get All Data")
    if st.button("Click to view whole Data"):
        st.dataframe(df)
