## main
import pandas as pd
import numpy as np
import os
import joblib

## skelarn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## Reading Data
DF_PATH = r"data/raw/explored_df.pkl"
TRAIN_PATH = os.path.join(os.getcwd(), DF_PATH)
df = pd.read_pickle(TRAIN_PATH)
df.reset_index(inplace=True, drop=True)

## 
cols = ['online_security', 'online_backup', 'device_protection_plan',
         'premium_tech_support', 'streaming_tv', 'streaming_movies','streaming_music']
DICT_REPLACE = {'Yes': 1, 'No': 0}
df_internet = df[cols].replace(DICT_REPLACE)
df_internet['enrolled_services'] = np.sum(df_internet, axis=1)
df['enrolled_services'] = df_internet['enrolled_services']

## To features and target
X_total = df.drop(columns=['customer_status', 'gender', 'enrolled_services'])
X = df.drop(columns=['customer_status', 'gender', 'online_security', 'online_backup', 'device_protection_plan',
       'premium_tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music'])
y = df['customer_status']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

## Slice the lists
num_cols = ['age', 'number_of_dependents', 'number_of_referrals', 'tenure_in_months', 'avg_monthly_long_distance_charges',
            'avg_monthly_gb_download', 'total_revenue', 'enrolled_services']

cat_cols = ['married', 'phone_service', 'multiple_lines', 'internet_service', 'unlimited_data', 'paperless_billing', 'offer', 'payment_method']

ord_cols_1 = ['internet_type']
ord_cols_2 = ['contract']



## Pipeline
## Numerical: num_cols --> Imputing using median, and standardscaler
## Categorical: cat_cols ---> Imputing using mode, and OHE
## Ordinal: ord_cols_1, ord_cols_2 ---> Imputing using mode, and ordinalEncoder

## For Numerical
num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', MinMaxScaler())
                    ])


## For Categorical
cat_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(cat_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ord_cols_1
ordinal_pipeline_1 = Pipeline(steps=[
                        ('selector', DataFrameSelector(ord_cols_1)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder',OrdinalEncoder(categories=[['No', 'Cable', 'DSL', 'Fiber Optic']]))
                    ])


## For ord_cols_1
ordinal_pipeline_2 = Pipeline(steps=[
                        ('selector', DataFrameSelector(ord_cols_2)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder',OrdinalEncoder(categories=[['Month-to-Month', 'One Year', 'Two Year']]))
                    ])


## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', cat_pipeline),
                                    ('ord_1', ordinal_pipeline_1),
                                    ('ord_2', ordinal_pipeline_2)
                                ])

## apply
all_pipeline.fit_transform(X_train)


## The Function to process new instances
def process_new(X_new):
    
    all_pipeline = joblib.load('models/pipeline.pkl')
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


    ## If you make feature engineering
    cols = ['online_security', 'online_backup', 'device_protection_plan',
         'premium_tech_support', 'streaming_tv', 'streaming_movies','streaming_music']
    DICT_REPLACE = {'Yes': 1, 'No': 0}
    df_internet = df[cols].replace(DICT_REPLACE)
    df_internet['enrolled_services'] = np.sum(df_internet, axis=1)
    df_new['enrolled_services'] = df_internet['enrolled_services']
    
    ## Call the pipeline
    X_processed = all_pipeline.transform(df_new)

    return X_processed