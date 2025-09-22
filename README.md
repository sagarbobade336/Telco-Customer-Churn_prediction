# Telco-Customer-Churn_prediction

**Telco Customer Churn Prediction**
This project deploys a machine learning model to predict customer churn for a telecommunications company. The application is built with Streamlit, providing an interactive and user-friendly interface for making real-time predictions based on customer data.

**Project Structure**
/Telco_Customer_Churn_Prediction/
├── app.py                     # The Streamlit application
├── best_model.pkl             # The pre-trained XGBoost model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # The original dataset
├── Telco Customer.ipynb       # Jupyter notebook with data analysis and model training
└── README.md                  # This file

**Features**

Interactive Web App: A simple and intuitive user interface built with Streamlit.

Predictive Model: Uses a pre-trained XGBoost Classifier to predict churn probability.

Data Preprocessing: Handles categorical and numerical features with scikit-learn's OneHotEncoder and StandardScaler to ensure consistency between training and prediction.

Actionable Insights: Provides a clear prediction result and a probability score, along with a recommendation for how to handle the customer.

**LINK**--> https://telco-customer-churnprediction-vpb5c96dplsf8joou8py2d.streamlit.app/
