import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from data_loader import load_kdd_data
from model import AnomalyDetector

# Page setup
st.set_page_config(page_title="KDD Anomaly Detector", layout="wide")
st.title("Network Attack Detection with ML")

# Load data
@st.cache_data
def load_data():
    X, y = load_kdd_data()
    return X, y

X, y = load_data()

# Sidebar controls
st.sidebar.header("Model Settings")
contamination = st.sidebar.slider(
    "Expected anomaly fraction", 
    0.01, 0.5, 0.05, 0.01
)

# Initialize and train model
detector = AnomalyDetector()
detector.model.set_params(contamination=contamination)

with st.spinner("Training model..."):
    detector.train(X)

# Make predictions
predictions = detector.predict(X)

# Convert predictions to match label format (1=attack, 0=normal)
pred_labels = np.where(predictions == -1, 1, 0)

# Results
st.subheader("Detection Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Actual Attacks", y.sum())
with col2:
    st.metric("Predicted Attacks", pred_labels.sum())

# Evaluation
accuracy = detector.evaluate(X, y)
st.metric("Accuracy", f"{accuracy:.1%}")

# Visualization
st.subheader("Feature Distribution")
feature = st.selectbox("Select feature", X.columns)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Normal vs Attack distribution
ax[0].hist(X[y == 0][feature], bins=50, alpha=0.7, label='Normal')
ax[0].hist(X[y == 1][feature], bins=50, alpha=0.7, label='Attack')
ax[0].set_title(f"{feature} distribution")
ax[0].legend()

# Predicted anomalies
ax[1].hist(X[pred_labels == 0][feature], bins=50, alpha=0.7, label='Normal')
ax[1].hist(X[pred_labels == 1][feature], bins=50, alpha=0.7, label='Anomaly')
ax[1].set_title(f"Model predictions")
ax[1].legend()

st.pyplot(fig)

# Show some examples
if st.checkbox("Show example anomalies"):
    examples = X.copy()
    examples['predicted_anomaly'] = pred_labels
    st.dataframe(
        examples[examples['predicted_anomaly'] == 1].head(100),
        height=300
    )