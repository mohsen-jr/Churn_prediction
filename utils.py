## main
import numpy as np
import pandas as pd
import joblib
import pickle



## Reading Data
DF_PATH = r"data/raw/explored_df.pkl"
def read_data(PATH):
    
    df = pd.read_pickle(PATH)
    df.reset_index(inplace=True, drop=True)
    return df


## feature engineering ---> create new column called "enrolled_services"
def feature_eng(df):
    cols = ['online_security', 'online_backup', 'device_protection_plan',
            'premium_tech_support', 'streaming_tv', 'streaming_movies','streaming_music']
    DICT_REPLACE = {'Yes': 1, 'No': 0}
    df_internet = df[cols].replace(DICT_REPLACE)
    df_internet['enrolled_services'] = np.sum(df_internet, axis=1)
    df['enrolled_services'] = df_internet['enrolled_services']
    return df


## To get features' names
df = read_data(DF_PATH)
df = feature_eng(df)
X_total = df.drop(columns=['customer_status', 'gender', 'enrolled_services'])

all_pipeline = joblib.load('models/pipeline.pkl')

## The Function to process new instances
def process_new(X_new):
    
    
    ## To DF
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_total.columns

    ## Adjust the dtypes of features
    df_new['age'] = df_new['age'].astype('int')
    df_new['married'] = df_new['married'].astype('str')
    df_new['number_of_dependents'] = df_new['number_of_dependents'].astype('int')
    df_new['number_of_referrals'] = df_new['number_of_referrals'].astype('float64')
    df_new['tenure_in_months'] = df_new['tenure_in_months'].astype('float64')
    df_new['offer'] = df_new['offer'].astype('str')
    df_new['phone_service'] = df_new['phone_service'].astype('str')
    df_new['avg_monthly_long_distance_charges'] = df_new['avg_monthly_long_distance_charges'].astype('float64')
    df_new['multiple_lines'] = df_new['multiple_lines'].astype('str')
    df_new['internet_service'] = df_new['internet_service'].astype('str')
    df_new['internet_type'] = df_new['internet_type'].astype('str')
    df_new['online_security'] = df_new['online_security'].astype('str')
    df_new['online_backup'] = df_new['online_backup'].astype('str')
    df_new['device_protection_plan'] = df_new['device_protection_plan'].astype('str')
    df_new['premium_tech_support'] = df_new['premium_tech_support'].astype('str')
    df_new['streaming_tv'] = df_new['streaming_tv'].astype('str')
    df_new['streaming_movies'] = df_new['streaming_movies'].astype('str')
    df_new['streaming_music'] = df_new['streaming_music'].astype('str')
    df_new['avg_monthly_gb_download'] = df_new['avg_monthly_gb_download'].astype('float64')
    df_new['unlimited_data'] = df_new['unlimited_data'].astype('str')
    df_new['contract'] = df_new['contract'].astype('str')
    df_new['paperless_billing'] = df_new['paperless_billing'].astype('str')
    df_new['payment_method'] = df_new['payment_method'].astype('str')
    df_new['total_revenue'] = df_new['total_revenue'].astype('float64')


    ## Feature engineering
    df_new = feature_eng(df_new)
    
    ## Call the pipeline
    X_processed = all_pipeline.transform(df_new)

    return X_processed