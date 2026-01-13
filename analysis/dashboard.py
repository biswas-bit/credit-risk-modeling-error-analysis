import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Credit Risk Error Analysis", layout="wide")

st.title("📊 Credit Risk Modeling: Model Comparison")
st.markdown("### Research Focus: Minimizing Financial Loss via High Recall")

# Load the data
df = pd.read_csv("model_comparison_results.csv")

# Metric Row
col1, col2, col3 = st.columns(3)
best_recall_model = df.loc[df['Recall (Default)'].idxmax()]

col1.metric("Highest Recall", best_recall_model['Model'], f"{best_recall_model['Recall (Default)']:.2%}")
col2.metric("Avg. Precision", f"{df['Precision (Default)'].mean():.2%}")
col3.metric("Total Models Tested", len(df))

# Visualizing the Trade-off (The Seesaw)
st.write("#### Precision vs. Recall Trade-off")
fig = px.bar(df, x='Model', y=['Precision (Default)', 'Recall (Default)'], 
             barmode='group', title="Precision vs. Recall per Model",
             color_discrete_sequence=['#EF553B', '#636EFA'])
st.plotly_chart(fig, use_container_width=True)

# Data Table
st.write("#### Detailed Performance Metrics")
st.dataframe(df.style.highlight_max(subset=['Recall (Default)', 'Accuracy'], color='lightgreen'))

st.info("💡 **Error Analysis Insight:** While XGBoost may have higher accuracy, the Balanced Random Forest is superior for risk mitigation due to its significantly higher Recall.")
