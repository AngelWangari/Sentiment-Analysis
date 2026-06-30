
import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mental Health Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🧠 Mental Health Sentiment Analysis Dashboard")
st.markdown(
    "Performance evaluation of Logistic Regression, Naive Bayes, "
    "Linear SVM and Hybrid Ensemble models."
)

# =====================================================
# DATASET INFORMATION
# =====================================================

TOTAL_RECORDS = 52674

class_distribution = pd.DataFrame({
    "Sentiment": [
        "Very Negative",
        "Neutral",
        "Negative"
    ],
    "Count": [
        26053,
        16339,
        10282
    ]
})

# =====================================================
# MODEL RESULTS
# =====================================================

results_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Linear SVM",
        "Naive Bayes",
        "LR + SVM Ensemble"
    ],
    "Accuracy": [
        0.8849,
        0.8846,
        0.7903,
        0.8960
    ],
    "Macro F1": [
        0.8732,
        0.8708,
        0.7679,
        0.8800
    ]
})

# =====================================================
# 1. OVERVIEW
# =====================================================

st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{TOTAL_RECORDS:,}")
c2.metric("Classes", "3")
c3.metric("Best Individual Model", "Logistic Regression")
c4.metric("Best Overall Model", "LR + SVM Ensemble")

st.markdown("---")

# =====================================================
# 2. CLASS DISTRIBUTION
# =====================================================

st.header("2. Sentiment Distribution")

fig = px.pie(
    class_distribution,
    names="Sentiment",
    values="Count",
    hole=0.45,
    title="Class Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================
# 3. HYPERPARAMETER TUNING
# =====================================================

st.header("3. Hyperparameter Tuning Results")

tuning_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Linear SVM"
    ],
    "Best Parameters": [
        "C=1, solver='lbfgs'",
        "alpha=0.1",
        "C=0.1"
    ]
})

st.dataframe(
    tuning_df,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# 4. MODEL PERFORMANCE
# =====================================================

st.header("4. Model Performance Summary")

st.dataframe(
    results_df,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# 5. ACCURACY COMPARISON
# =====================================================

st.header("5. Accuracy Comparison")

fig_acc = px.bar(
    results_df,
    x="Model",
    y="Accuracy",
    text="Accuracy",
    title="Accuracy Scores"
)

fig_acc.update_layout(yaxis_range=[0, 1])

fig_acc.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

st.plotly_chart(
    fig_acc,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# 6. MACRO F1 COMPARISON
# =====================================================

st.header("6. Macro F1 Comparison")

fig_f1 = px.bar(
    results_df,
    x="Model",
    y="Macro F1",
    text="Macro F1",
    title="Macro F1 Scores"
)

fig_f1.update_layout(yaxis_range=[0, 1])

fig_f1.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

st.plotly_chart(
    fig_f1,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# 7. DETAILED CLASSIFICATION REPORTS
# =====================================================

st.header("7. Detailed Classification Reports")

reports = {

    "Logistic Regression": {
        "Accuracy": "88.49%",
        "Negative F1": "0.82",
        "Neutral F1": "0.90",
        "Very Negative F1": "0.90",
        "Insight": "Best individual model after GridSearchCV tuning."
    },

    "Linear SVM": {
        "Accuracy": "88.46%",
        "Negative F1": "0.81",
        "Neutral F1": "0.90",
        "Very Negative F1": "0.91",
        "Insight": "Nearly identical performance to Logistic Regression."
    },

    "Naive Bayes": {
        "Accuracy": "79.03%",
        "Negative F1": "0.72",
        "Neutral F1": "0.75",
        "Very Negative F1": "0.83",
        "Insight": "Lowest performance but useful diversity contributor."
    },

    "LR + SVM Ensemble": {
        "Accuracy": "89.60%",
        "Negative F1": "0.82",
        "Neutral F1": "0.91",
        "Very Negative F1": "0.91",
        "Insight": "Highest overall accuracy and strongest generalization."
    }
}

for model, values in reports.items():

    st.subheader(model)

    st.write(
        f"**Accuracy:** {values['Accuracy']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Negative F1",
        values["Negative F1"]
    )

    col2.metric(
        "Neutral F1",
        values["Neutral F1"]
    )

    col3.metric(
        "Very Negative F1",
        values["Very Negative F1"]
    )

    st.info(values["Insight"])

    st.markdown("---")

# =====================================================
# 8. BEST MODEL
# =====================================================

st.header("8. Best Model")

st.success(
    """
    LR + SVM Ensemble achieved the highest performance.

    Accuracy : 89.60%

    Macro F1 : 88.00%

    The ensemble combines Logistic Regression and
    Linear SVM predictions through majority voting,
    resulting in improved generalization and
    better sentiment classification performance.
    """
)

# =====================================================
# 9. RESEARCH CONCLUSION
# =====================================================

st.header("9. Research Conclusion")

st.info(
    """
    Hyperparameter tuning using GridSearchCV improved
    the performance of all machine learning models.

    Logistic Regression emerged as the strongest
    standalone classifier with 88.49% accuracy.

    The LR + SVM Ensemble further improved performance
    to 89.60% accuracy, making it the recommended model
    for deployment in the Mental Health Sentiment
    Analysis System.
    """
)

st.markdown("---")
st.caption("Mental Health Sentiment Analysis Project Dashboard")
