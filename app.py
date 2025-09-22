import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define the columns used in the original training data
NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

# Load the dataset to fit the preprocessors
# This is a crucial step to ensure the one-hot encoding is consistent
# with the training data.
try:
    df_train = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'], errors='coerce')
    df_train.fillna(0, inplace=True)

    # Initialize and fit the OneHotEncoder on the training data
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df_train[CAT_COLS])

    # Initialize and fit the StandardScaler on the numerical training data
    scaler = StandardScaler()
    scaler.fit(df_train[NUM_COLS])

except FileNotFoundError:
    st.error("Error: Dataset file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    st.stop()

# Function to load the model and preprocessors
@st.cache_resource
def load_resources():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'best_model.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

# Function to preprocess user input
def preprocess_input(input_df):
    
    # Preprocess numerical columns
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df.fillna(0, inplace=True)
    
    numerical_data = scaler.transform(input_df[NUM_COLS])
    numerical_df = pd.DataFrame(numerical_data, columns=[f'{col}_scaled' for col in NUM_COLS])
    
    # Preprocess categorical columns using the pre-fitted OneHotEncoder
    categorical_data = ohe.transform(input_df[CAT_COLS])
    feature_names = ohe.get_feature_names_out(CAT_COLS)
    categorical_df = pd.DataFrame(categorical_data, columns=feature_names)

    # Combine the preprocessed data
    preprocessed_df = pd.concat([numerical_df, categorical_df], axis=1)
    
    # The model expects a specific order of columns.
    # The following code re-orders the columns to match the model's expectations.
    
    # Get the feature names from the fitted model for correct ordering
    model_feature_names = model.feature_names_in_
    
    # Create a new DataFrame with the correct column order
    final_df = preprocessed_df.reindex(columns=model_feature_names, fill_value=0)
    
    return final_df

# Load the model
model = load_resources()

# Streamlit App
st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")

st.title("Telco Customer Churn Prediction")
st.markdown("### Enter Customer Details")

# Input form
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        SeniorCitizen = st.radio("Senior Citizen", ["Yes", "No"])
        Partner = st.radio("Partner", ["Yes", "No"])
        Dependents = st.radio("Dependents", ["Yes", "No"])
    
    with col2:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=600.0)
        
    with col3:
        PhoneService = st.radio("Phone Service", ["Yes", "No"])
        MultipleLines = st.radio("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.radio("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.radio("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.radio("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.radio("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.radio("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.radio("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.radio("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create a DataFrame from the user inputs
    user_input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Preprocess the input
    preprocessed_data = preprocess_input(user_input_df)

    # Make prediction
    prediction_proba = model.predict_proba(preprocessed_data)[:, 1]
    prediction = model.predict(preprocessed_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Prediction: The customer is likely to churn. (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Prediction: The customer is unlikely to churn. (Probability: {prediction_proba[0]:.2f})")

    st.write("---")
    st.subheader("Input Features")
    st.dataframe(user_input_df)
    st.write("---")
    st.subheader("Preprocessed Data for Model")
    st.dataframe(preprocessed_data)