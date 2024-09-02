import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

lof_model = joblib.load('LOF_model.pkl')

def classify_transactions(model, new_data):
    # first preprocessing and scaling the data like the data used in the notebook
    remove_non_numbers = new_data.applymap(lambda x: isinstance(x, str))
    new_data = new_data[~remove_non_numbers.any(axis=1)]    
    # scaling 
    scaler = MinMaxScaler()
    new_data['Scaled_amt'] = scaler.fit_transform(new_data[['Amount']])
    scaler = StandardScaler()
    new_data['Scaled_time'] = scaler.fit_transform(new_data[['Time']])

    # PCA
    feature_df = new_data.drop(columns=['Class', 'Amount', 'Time'])
    pca = PCA(n_components=2)
    pca_new = pca.fit_transform(feature_df)
    pca_new_dim_2 = pd.DataFrame(data=pca_new, columns=['PCA1', 'PCA2'])
    pca_new_dim_2['Class'] = new_data['Class']
    pca_new_dim_2 = pca_new_dim_2.dropna(subset=['Class'])
    
    predictions = lof_model.predict(pca_new_dim_2[['PCA1', 'PCA2']])
    # convert predictions to class
    predictions = [1 if x == -1 else 0 for x in predictions]
    
    # add the predictions as a column in original dataset
    new_data['Fraudulent'] = predictions
    
    # filter out the fraudulent transactions for displaying to user
    fraud_detected = new_data[new_data['Fraudulent'] == 1]
    
    return fraud_detected

st.title("Credit Card Fraud Detection")

file_input = st.file_uploader("Upload transactions to detect frauds", type=["csv", "xlsx"])

if file_input is not None:
    # Read the uploaded file
    if file_input.name.endswith('.csv'):
        user_transactions = pd.read_csv(file_input)
    else:
        user_transactions = pd.read_excel(file_input, sheet_name='creditcard_test')

    st.write("Data Preview:")
    st.dataframe(user_transactions.head())

    fraud_transactions = classify_transactions(lof_model, user_transactions)
    
    st.write(f"Number of fraudulent transactions detected: {len(fraud_transactions)}")
    
    st.write("Fraudulent Transactions:")
    st.dataframe(fraud_transactions)

    st.write("Visualizing Anomalies:")
    if len(fraud_transactions) > 0:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=fraud_transactions, x='PCA1', y='PCA2', hue='Fraudulent', palette='Reds')
        plt.title('Detected Anomalies')
        st.pyplot(plt)
    else:
        st.write("No anomalies detected.")