import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render(df):
    st.header("💡 Pearson Correlation Matrix")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr().round(2)

    hovertext = [[f"{col1} vs {col2}: {corr.loc[col1, col2]}" for col2 in corr.columns] for col1 in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0.0, "#d73027"], [0.5, "#ffffbf"], [1.0, "#1a9850"]],
        zmin=-1, zmax=1,
        hoverinfo="text", text=hovertext,
        colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1])
    ))

    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            val = corr.loc[row, col]
            font_color = "black" if abs(val) < 0.6 else "white"
            fig.add_annotation(x=col, y=row, text=str(val), showarrow=False,
                               font=dict(size=11, color=font_color))

    fig.update_layout(
        title="💡 Pearson Correlation Matrix", title_x=0.5,
        width=900, height=700, font=dict(size=14),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strongest Correlations")
    corr_unstacked = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    corr_series = corr_unstacked.unstack().dropna().sort_values(key=lambda x: abs(x), ascending=False)
    top_corr = corr_series.reset_index()
    top_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']

    def strength_label(val):
        abs_val = abs(val)
        if abs_val >= 0.7:
            return "Strong"
        elif abs_val >= 0.4:
            return "Moderate"
        elif abs_val >= 0.2:
            return "Weak"
        return "Very Weak"

    top_corr['Strength'] = top_corr['Correlation'].apply(strength_label)
    st.dataframe(top_corr.head(15), use_container_width=True)
