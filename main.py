import joblib
import numpy as np
import streamlit as st
from utils import process_new


## Loading the Model
model = joblib.load('models/XGBoost_tuned.pkl')



def churn_classification():
    ## Streamlit
    st.title('Customer Churn Classification ..')
    st.divider()

    ## Input Fields
    age = st.number_input('Age', value=40)
    married = st.selectbox('Married', options=['Yes', 'No'])
    num_of_dependents = st.slider('Number of dependents', min_value=0, max_value=10, step=1)
    num_of_referals = st.slider('Number of referals', min_value=0, max_value=10, step=1)
    tenure = st.slider('Tenure', min_value=0, max_value=100, step=1)
    offer = st.selectbox('Offer', options=['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'None'])
    phone_service = st.selectbox('Phone Service', options=['Yes', 'No'])
    avg_monthly_long_distance_charges = st.number_input('Avg. Monthly Long Distance Charges', value=0)
    multiple_lines = st.selectbox('Multiple Lines', options=['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', options=['Yes', 'No'])
    internet_type = st.selectbox('Internet Type', options=['Cable', 'DSL', 'Fiber Optic', 'No'])
    online_security = st.selectbox('Online Security', options=['Yes', 'No'])
    online_backup = st.selectbox('Online Backup', options=['Yes', 'No'])
    device_protection_plan = st.selectbox('Device Protection Plan', options=['Yes', 'No'])
    premium_tech_support = st.selectbox('Premium Tech Support', options=['Yes', 'No'])
    streaming_tv = st.selectbox('Streaming TV', options=['Yes', 'No'])
    streaming_movies = st.selectbox('Streaming Movies', options=['Yes', 'No'])
    streaming_music = st.selectbox('Streaming Music', options=['Yes', 'No'])
    avg_monthly_gb_download = st.number_input('Avg Monthly GB Download', value=0)
    unlimited_data = st.selectbox('Unlimited Data', options=['Yes', 'No'])
    contract = st.selectbox('Contract', options=['Month-to-Month', 'One Year', 'Two Year'])
    paperless_billing = st.selectbox('Paperless Billing', options=['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', options=['Mailed Check', 'Credit Card', 'Bank Withdrawal'])
    total_revenue = st.number_input('Total Revenue', value=0)


    if st.button('Predict Churn ...'):

        ## Concatenate features
        new_data = np.array([age, married, num_of_dependents, num_of_referals, tenure, offer, phone_service, 
                                avg_monthly_long_distance_charges, multiple_lines, internet_service, internet_type, avg_monthly_gb_download,
                                  online_security, online_backup, device_protection_plan, premium_tech_support, streaming_tv,
                                    streaming_movies, streaming_music, unlimited_data,
                                      contract, paperless_billing, payment_method, total_revenue])
        
        ## Call the function (process_new) from utils.py
        X_prcessed = process_new(X_new=new_data)

        ## Model prediction
        y_pred = model.predict(X_prcessed)[0]

        ## To True or False
        y_pred = bool(y_pred)

        st.success(f'Churn prediction is: {y_pred}')



## Run via Terminal
if __name__ == '__main__':
    ## Call the above function
    churn_classification()