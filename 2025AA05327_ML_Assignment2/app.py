import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set page config
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

st.title("Machine Learning Classification Dashboard")
st.markdown("### M.Tech (AIML/DSE) - Assignment 2")
st.markdown("💳 **Dataset:** Default of Credit Card Clients (UCI ID 42477) - Sampled 50% of the datasets")
st.info("More Details - https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")

# --- 1. Data Loading (Fixed to Default 50% Sample) ---
@st.cache_data
def load_and_prep_data():
    """
    Loads the Default of Credit Card Clients dataset (ID 42477),
    Samples 50%, and prepares data.
    """
    try:
        # Fetch data
        data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # Sampling 50% of the dataset for using a smaller dataset
        X = X.sample(frac=0.5, random_state=42)
        y = y.loc[X.index]

        # Ensure target is numeric
        if y.dtype == 'category' or y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, pd.Series(y)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

with st.spinner("Loading dataset ..."):
    X_full, y_full = load_and_prep_data()

if X_full is None:
    st.stop()

# --- 2. Train/Test Split ---
# We keep a fixed internal split. 
# 80% of the 50% sample is used for training.
# 20% of the 50% sample is kept as the "Internal Test Set".
X_train, X_internal_test, y_train, y_internal_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Scaling - Fit on TRAIN only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Sidebar for static info only
st.sidebar.header("Dataset Info")
st.sidebar.success(f"Total Records: {len(X_full)}")
st.sidebar.info(f"Training Set: {len(X_train)}")
st.sidebar.warning(f"Test Set: {len(X_internal_test)}")

st.divider()

# --- 3. Model Selection (Main Page) ---
st.subheader("1. Model Configuration & Training")

col_model_select, col_model_params = st.columns(2)

with col_model_select:
    model_choice = st.selectbox(
        "Choose Classification Model",
        [
            "Logistic Regression",
            "Decision Tree Classifier",
            "K-Nearest Neighbor (KNN)",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ]
    )

# Initialize Model & Params
model = None
with col_model_params:
    if model_choice == "Logistic Regression":
        st.write("Hyperparameters: Default (Max Iter=1000)")
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree Classifier":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_choice == "K-Nearest Neighbor (KNN)":
        n_neighbors = st.slider("Neighbors (K)", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_choice == "Naive Bayes (Gaussian)":
        st.write("Hyperparameters: Default")
        model = GaussianNB()
    elif model_choice == "Random Forest (Ensemble)":
        n_estimators = st.slider("Trees", 5, 100, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "XGBoost (Ensemble)":
        st.write("Hyperparameters: Default (LogLoss)")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train Model Button (Main Page)
if st.button("Train Model", type="primary"):
    with st.spinner(f"Training {model_choice}..."):
        # Train on scaled data for LR/KNN, raw for others
        if model_choice in ["Logistic Regression", "K-Nearest Neighbor (KNN)"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
            
    # Save model and choice to session state
    st.session_state['trained_model'] = model
    st.session_state['model_name'] = model_choice
    st.session_state['scaler'] = scaler # Save scaler for test data
    st.success(f"✅ {model_choice} Trained Successfully!")

st.divider()

# --- 4. Test Data Management (Download & Upload) ---
st.subheader("2. Test Data Management")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Step A: Download Test Data")
    st.write("Download the unseen test set (CSV) to your local machine.")
    
    # Prepare CSV for download (Feature + Target)
    test_download_df = X_internal_test.copy()
    test_download_df['target'] = y_internal_test.values  # Add target column
    csv_data = test_download_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Test Data (CSV)",
        data=csv_data,
        file_name="test_data_credit_card.csv",
        mime="text/csv",
    )

with col2:
    st.markdown("#### Step B: Upload Test Data")
    st.write("Upload the CSV file to evaluate the trained model.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

st.divider()

# --- 5. Evaluation Logic ---
st.subheader("3. Evaluation Results")

if 'trained_model' in st.session_state:
    model = st.session_state['trained_model']
    model_name = st.session_state['model_name']
    saved_scaler = st.session_state['scaler']
    
    X_eval, y_eval = None, None
    source_msg = ""

    # Logic: If file uploaded, use it. Else, use internal test set.
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            # separate target
            # assume target is the last column or named 'target'
            if 'target' in df_upload.columns:
                target_col = 'target'
            else:
                target_col = df_upload.columns[-1]
                
            X_eval = df_upload.drop(columns=[target_col])
            y_eval = df_upload[target_col]
            
            # Align columns
            X_eval = X_eval[X_train.columns]
            source_msg = "Evaluated on: **Uploaded File**"
            
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()
    else:
        # Fallback if no file uploaded yet
        X_eval = X_internal_test
        y_eval = y_internal_test
        source_msg = "Evaluated on: **Internal Test Set** (Upload a file to change)"

    st.markdown(source_msg)

    # Prediction
    # Apply scaling if needed
    if model_name in ["Logistic Regression", "K-Nearest Neighbor (KNN)"]:
        X_eval_ready = saved_scaler.transform(X_eval)
    else:
        X_eval_ready = X_eval

    y_pred = model.predict(X_eval_ready)
    y_prob = model.predict_proba(X_eval_ready)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate Metrics
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_prob) if y_prob is not None else "N/A"

    # Display Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC", f"{mcc:.4f}")

    # Plots
    st.markdown("---")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_eval, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    with col_plot2:
        st.markdown("**Classification Report**")
        report = classification_report(y_eval, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("👈 Please select a model and click 'Train Model' in Section 1 to start.")
