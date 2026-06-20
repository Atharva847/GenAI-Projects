import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(df):
    st.markdown("### 📊 Descriptive Statistics")
    st.dataframe(df.describe().transpose(), use_container_width=True)
    st.markdown("---")

    st.markdown("### ❗ Missing Values Summary")
    null_percent = (df.isnull().mean() * 100).round(2)
    null_percent = null_percent[null_percent > 0]

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.dataframe(df.isnull().sum().rename("Missing Count"), use_container_width=True)
    with c2:
        if not null_percent.empty:
            fig = px.pie(
                names=null_percent.index,
                values=null_percent.values,
                title="Missing Value (%)",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textinfo='label+percent', pull=[0.05] * len(null_percent))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found.")

    st.markdown("---")
    num_df = df.select_dtypes(include=['int', 'float'])
    st.markdown("### 🧮 Skewness, Kurtosis & Outlier Count")
    sk_df = pd.DataFrame({
        "Mean": num_df.mean(),
        "Std Dev": num_df.std(),
        "Skewness": num_df.skew(),
        "Kurtosis": num_df.kurtosis(),
        "Outliers (IQR Method)": [
            ((num_df[col] < (num_df[col].quantile(0.25) - 1.5 * (num_df[col].quantile(0.75) - num_df[col].quantile(0.25)))) |
             (num_df[col] > (num_df[col].quantile(0.75) + 1.5 * (num_df[col].quantile(0.75) - num_df[col].quantile(0.25))))
             ).sum()
            for col in num_df.columns
        ]
    }).round(3)
    st.dataframe(sk_df, use_container_width=True)

    st.markdown("### 📉 Skewness & Kurtosis Plot")
    fig = px.bar(
        sk_df.reset_index(),
        x='index',
        y=["Skewness", "Kurtosis"],
        barmode='group',
        title="Skewness and Kurtosis by Numeric Column",
        labels={"index": "Column"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Histogram with Outliers (IQR)")
    selected_hist_col = st.selectbox("Select column for histogram with outliers", num_df.columns)
    if selected_hist_col:
        col_data = num_df[selected_hist_col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower) | (col_data > upper)]
        normal = col_data[(col_data >= lower) & (col_data <= upper)]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=normal, name='Normal', marker_color='green', opacity=0.75))
        fig.add_trace(go.Histogram(x=outliers, name='Outliers', marker_color='red', opacity=0.75))
        fig.update_layout(
            title=f"Histogram of {selected_hist_col} with Outliers Highlighted",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
