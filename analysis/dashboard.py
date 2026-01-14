import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Credit Risk Error Analysis 2026", layout="wide")

# 2. Load Data
@st.cache_data
def load_data():
    # Using your updated summary file
    data = pd.read_csv("model_summary.csv")
    return data

df = load_data()

# 3. Header & Research Context
st.title("📊 Credit Risk Modeling & Error Analysis")
st.markdown("""
**Research Objective:** Evaluating top ML algorithms on imbalanced datasets to minimize 
**Total Financial Loss** (High Recall focus).
""")

# 4. Top Level Metrics (KPIs)
col1, col2, col3, col4 = st.columns(4)
best_recall_val = df['Recall (Default)'].max()
best_recall_model = df.loc[df['Recall (Default)'].idxmax()]['Model']

col1.metric("🏆 Highest Recall", f"{best_recall_val:.2%}", best_recall_model)
col2.metric("🎯 Avg. Precision", f"{df['Precision (Default)'].mean():.2%}")
col3.metric("📉 Min. Threshold", f"{df['Threshold'].min()}")
col4.metric("📚 Models Analyzed", len(df))

st.divider()

# 5. Visualizations
row1_col1, row1_col2 = st.columns([1.5, 1])

with row1_col1:
    st.subheader("The 'Seesaw' Analysis: Precision vs. Recall")
    # Melt dataframe for better Plotly grouping
    df_melted = df.melt(id_vars="Model", value_vars=['Precision (Default)', 'Recall (Default)'], 
                        var_name="Metric", value_name="Value")
    
    fig_bar = px.bar(df_melted, x='Model', y='Value', color='Metric', barmode='group',
                     color_discrete_map={'Precision (Default)': '#EF553B', 'Recall (Default)': '#636EFA'},
                     text_auto='.2f')
    st.plotly_chart(fig_bar, use_container_width=True)

with row1_col2:
    st.subheader("Model Error Distribution")
    # Radar-style or Bubble chart to show balance
    fig_scatter = px.scatter(df, x="Precision (Default)", y="Recall (Default)",
                             size="F1-Score (Default)", color="Model",
                             hover_data=['Threshold', 'Accuracy'],
                             title="Precision-Recall Space (Size = F1 Score)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# 6. Detailed Data Table
st.subheader("Detailed Performance Metrics")
st.dataframe(df.style.highlight_max(subset=['Recall (Default)'], color='#2E7D32')
                     .highlight_min(subset=['Threshold'], color='#1565C0'), 
             use_container_width=True)

# 7. Research Conclusion: Why Balanced Random Forest Won?
st.divider()
st.subheader("📝 Research Findings: The Balanced Random Forest Performance")

# Create a clean layout for the explanation
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.markdown("""
    ### Why it outperformed on Recall:
    *   **Bootstrap Under-sampling:** Unlike XGBoost which uses weights on the whole set, **Balanced RF** creates hundreds of balanced sub-datasets. 
    *   **Minority Dominance:** Each individual tree is forced to learn patterns of 'Defaults' without being overwhelmed by 'Non-Defaults'.
    *   **Majority Voting:** The final ensemble aggregates these 'specialized' trees, making it much more sensitive to risk signals.
    """)

with exp_col2:
    st.info("""
    **Financial Impact Analysis:**
    In this research, a **False Negative** (missing a default) is significantly more costly than a **False Positive** (wrongly flagging a good client). 
    
    The Balanced Random Forest's **Recall of 68%** suggests that it catches nearly **3x more defaults** than the baseline XGBoost (Recall 23-26%), making it the most economically viable choice for risk-averse lenders.
    """)

st.caption("Generated for Credit Risk Modeling Research - Jan 2026")
