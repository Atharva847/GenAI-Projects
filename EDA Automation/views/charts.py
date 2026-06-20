import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def render(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader("📈 Interactive Visualizations")
    chart_type = st.selectbox("Select a chart type", [
        "Histogram + KDE", "Scatter Plot", "Box Plot", "Bar Chart",
        "Pair Plot", "Histogram with Facets", "Swarm Plot", "Donut Chart",
        "Stacked Bar Chart", "Line Plot"
    ])

    if chart_type == "Histogram + KDE":
        if numeric_cols:
            selected_col = st.selectbox("Select a numeric column", numeric_cols, key="hist_col")
            col_data = df[selected_col].dropna()
            if len(col_data) > 5000:
                col_data = col_data.sample(5000, random_state=42)
            bin_size = (col_data.max() - col_data.min()) / 30 if col_data.nunique() > 10 else 1
            fig = ff.create_distplot([col_data], [selected_col], bin_size=bin_size, show_rug=False)
            fig.update_layout(title_text=f'Distribution Plot for {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-Axis", numeric_cols, key="scatter_x")
            y = st.selectbox("Y-Axis", numeric_cols, index=1, key="scatter_y")
            fig = px.scatter(df, x=x, y=y, title=f'Scatter Plot: {x} vs {y}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns.")

    elif chart_type == "Box Plot":
        if numeric_cols and categorical_cols:
            y = st.selectbox("Y-Axis (Numeric)", numeric_cols, key="box_y")
            x = st.selectbox("X-Axis (Categorical)", categorical_cols, key="box_x")
            fig = px.box(df, x=x, y=y, title=f'Box Plot: {y} by {x}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least one numeric and one categorical column.")

    elif chart_type == "Bar Chart":
        if categorical_cols:
            x = st.selectbox("X-Axis (Categorical)", categorical_cols, key="bar_x")
            count_df = df[x].value_counts().reset_index()
            count_df.columns = [x, "Count"]
            fig = px.bar(count_df, x=x, y="Count", title=f'Bar Chart: Count of {x}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns to plot.")

    elif chart_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            st.warning("Sampling top 5 numeric columns for performance.")
            sampled_df = df[numeric_cols].dropna().sample(
                n=500 if len(df) > 500 else len(df), random_state=42
            )
            fig = sns.pairplot(sampled_df[numeric_cols[:5]])
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns.")

    elif chart_type == "Histogram with Facets":
        if numeric_cols and categorical_cols:
            num = st.selectbox("Select numeric column", numeric_cols, key="facet_num")
            cat = st.selectbox("Select categorical column", categorical_cols, key="facet_cat")
            unique_vals = df[cat].nunique()
            if unique_vals > 25:
                st.warning(f"⚠️ {cat} has {unique_vals} unique values — too many for facet plot.")
            else:
                try:
                    fig = px.histogram(df, x=num, facet_col=cat,
                                       title=f"Histogram of {num} by {cat}", facet_col_spacing=0.04)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Failed to render facet histogram: {str(e)}")
        else:
            st.warning("Requires at least one numeric and one categorical column.")

    elif chart_type == "Swarm Plot":
        if numeric_cols and categorical_cols:
            num = st.selectbox("Y-Axis (Numeric)", numeric_cols, key="swarm_num")
            cat = st.selectbox("X-Axis (Categorical)", categorical_cols, key="swarm_cat")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.swarmplot(x=cat, y=num, data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Requires both numeric and categorical columns.")

    elif chart_type == "Donut Chart":
        if categorical_cols:
            cat = st.selectbox("Select categorical column", categorical_cols, key="donut_cat")
            donut_data = df[cat].value_counts()
            fig = go.Figure(data=[go.Pie(labels=donut_data.index, values=donut_data.values, hole=0.4)])
            fig.update_layout(title=f"Donut Chart: {cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available.")

    elif chart_type == "Stacked Bar Chart":
        if len(categorical_cols) >= 2:
            col1 = st.selectbox("X-Axis", categorical_cols, key="stack_x")
            col2 = st.selectbox("Color Category", categorical_cols, key="stack_color")
            if col1 == col2:
                st.warning("Please select different columns for X-Axis and Color Category.")
            else:
                stacked_df = df.groupby([col1, col2]).size().to_frame('Count').reset_index()
                fig = px.bar(stacked_df, x=col1, y='Count', color=col2,
                             title=f'Stacked Bar Chart: {col1} by {col2}')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 categorical columns.")

    elif chart_type == "Line Plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-Axis", numeric_cols, key="line_x")
            y = st.selectbox("Y-Axis", numeric_cols, index=1, key="line_y")
            fig = px.line(df, x=x, y=y, title=f'Line Plot: {y} over {x}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns.")
