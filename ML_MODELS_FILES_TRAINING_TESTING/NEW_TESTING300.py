import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime, timedelta
import os
import glob

# Ensure the predictions directory exists
os.makedirs('predictions', exist_ok=True)

# Connect to MongoDB servers
data_client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
save_client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')

data_db = data_client['intelliinvest']
save_db = save_client['intelliinvest']

fundamentals_collection = data_db['STOCK_FUNDAMENTALS']
signals_collection = data_db['STOCK_SIGNALS_COMPONENTS_10']
details_collection = data_db['STOCK_DETAILS']
predictions_collection = save_db['STOCK_PREDICTIONS']

# Function to calculate RMSE
def calculate_rmse(true_values, predictions):
    return np.sqrt(mean_squared_error(true_values, predictions))

# Function to calculate percentage error
def calculate_percentage_error(true_values, predictions):
    return np.abs((true_values - predictions) / true_values) * 100

# Retrieve all securityIds
fundamentals_security_ids = fundamentals_collection.distinct('securityId')
signals_security_ids = signals_collection.distinct('securityId')
details_security_ids = details_collection.distinct('securityId')
all_security_ids = set(fundamentals_security_ids).intersection(set(signals_security_ids), set(details_security_ids))

# Retrieve unique industries
industries = fundamentals_collection.distinct('industry')

print("-"*50)
print("Industries: ", industries)
print("-"*50)

# Define periods for predictions
periods = {
    'daily': 1,
    'weekly': 7,
    'monthly': 30,
    'quarterly': 90
}

# Load models for selected stocks in each industry
industry_models = {}

# Define the model types
model_types = ['lr', 'dt', 'gb']

# Iterate over each industry and model type to load models
for industry in industries:
    for model_type in model_types:
        # Create the pattern to match files with the _{industry}.pkl suffix for the current model type
        pattern = f'models/{model_type}model{industry}_*.pkl'

        # Find files that match the pattern
        matched_files = glob.glob(pattern)

        # Check if any files matched
        if matched_files:
            # Initialize a dictionary to hold models for each risk category in this industry and model type
            if industry not in industry_models:
                industry_models[industry] = {'lr': {}, 'dt': {}, 'gb': {}}

            # Open and load each matched file
            for file_to_open in matched_files:
                try:
                    with open(file_to_open, 'rb') as f:
                        model = pickle.load(f)
                        # Extract the risk category from the filename
                        risk_category = file_to_open.split('_')[-1].replace('.pkl', '')
                        industry_models[industry][model_type][risk_category] = model
                        print(f"Loaded {model_type} model for {industry} - {risk_category} risk: {file_to_open}")
                except Exception as e:
                    print(f"Failed to load model: {file_to_open}, error: {e}")
        else:
            print(f"No files found matching pattern: {pattern}")

print("-"*50)
print("-"*50)

# Function to assign risk category based on stdDevOfReturn
def assign_risk_category(row):
    if row['stdDevOfReturn'] >= 0 and row['stdDevOfReturn'] <= 0.25:
        return 'low'
    elif row['stdDevOfReturn'] > 0.25 and row['stdDevOfReturn'] <= 0.45:
        return 'medium'
    else:
        return 'high'

# Make predictions for each securityId
for security_id in all_security_ids:
    query = {"securityId": security_id}
    fundamentals_results = fundamentals_collection.find(query)
    signals_results = signals_collection.find(query)
    details_results = details_collection.find(query)

    fundamentals_df = pd.DataFrame(list(fundamentals_results))
    signals_df = pd.DataFrame(list(signals_results))
    details_df = pd.DataFrame(list(details_results))

    fundamentals_df = fundamentals_df.rename(columns={'todayDate': 'signalDate'})
    details_df = details_df.rename(columns={'todayDate': 'signalDate'})
    fundamentals_df['signalDate'] = pd.to_datetime(fundamentals_df['signalDate'])
    signals_df['signalDate'] = pd.to_datetime(signals_df['signalDate'])

    details_df = details_df[['securityId', 'signalDate', 'stdDevOfReturn']]
    details_df['signalDate'] = pd.to_datetime(details_df['signalDate'])

    merged_data = pd.merge(fundamentals_df, signals_df, on=['securityId', 'signalDate'], how='inner')
    merged_data = pd.merge(merged_data, details_df, on=['securityId', 'signalDate'], how='inner')

    selected_columns = ['signalDate', 'closePrice', 'securityId', 'alMarketCap', 'stdDevOfReturn', 'industry', 'TRn', 'ADXn', 'high10Day', 'low10Day', 'stochastic10Day', 'range10Day', 'percentKFlow', 'percentDFlow', 'upperBound', 'lowerBound', 'bandwidth', 'movingAverage_5', 'movingAverage_10', 'movingAverage_15', 'movingAverage_25', 'movingAverage_50']
    df = merged_data[selected_columns]

    df.set_index('securityId', inplace=True)

    df = df.sort_values(by='signalDate')

    df['category'] = df.apply(assign_risk_category, axis=1)

    # Convert signalDate to float (timestamp)
    df['signalDate'] = df['signalDate'].apply(lambda x: x.timestamp())

    test_df = df[df['signalDate'] == df['signalDate'].max()]

    if len(test_df) == 0:
        print(f"No test data available for securityId: {security_id}, skipping.")
        continue

    X_test = test_df.drop(columns=['closePrice', 'category', 'industry'])
    y_test = test_df['closePrice']

    industry = test_df['industry'].iloc[0]
    risk_category = test_df['category'].iloc[0]

    # Check if models exist for the industry and risk category
    if industry not in industry_models or risk_category not in industry_models[industry]['lr']:
        print(f"No models found for industry: {industry}, risk category: {risk_category}, skipping predictions for securityId: {security_id}.")
        continue

    models = {
        'lr': industry_models[industry]['lr'][risk_category],
        'dt': industry_models[industry]['dt'][risk_category],
        'gb': industry_models[industry]['gb'][risk_category]
    }

    last_date = pd.to_datetime(df['signalDate'].max(), unit='s')

    for period, days in periods.items():
        date = last_date + timedelta(days=days)

        if period == 'daily':
            lagged_date = last_date
        elif period == 'weekly':
            lagged_date = last_date - timedelta(days=7)
        elif period == 'monthly':
            lagged_date = last_date - timedelta(days=30)
        elif period == 'quarterly':
            lagged_date = last_date - timedelta(days=90)

        lagged_data = df[df['signalDate'] == lagged_date.timestamp()]

        if len(lagged_data) == 0:
            print(f"No historical data available for securityId: {security_id}, period: {period}, skipping.")
            continue

        new_data = lagged_data.iloc[-1:].copy()
        new_data['signalDate'] = date.timestamp()
        new_data = new_data.drop(columns=['closePrice'])

        X_new = new_data.drop(columns=['category', 'industry'])

        # Make predictions using the models for the matching risk category
        lr_predictions = models['lr'].predict(X_new)
        dt_predictions = models['dt'].predict(X_new)
        gb_predictions = models['gb'].predict(X_new)

        prediction_doc = {
            'securityId': security_id,
            'signalDate': date,
            'period': period,
            'Bagging_LR_Prediction': lr_predictions[0],
            'Bagging_DT_Prediction': dt_predictions[0],
            'Bagging_GB_Prediction': gb_predictions[0],
            'Actual_Price': df['closePrice'].iloc[-1],
            'Bagging_LR_Percentage_Error': calculate_percentage_error(df['closePrice'].iloc[-1], lr_predictions[0]),
            'Bagging_DT_Percentage_Error': calculate_percentage_error(df['closePrice'].iloc[-1], dt_predictions[0]),
            'Bagging_GB_Percentage_Error': calculate_percentage_error(df['closePrice'].iloc[-1], gb_predictions[0]),
            'Data_Category': df['category'].iloc[-1]
        }

        # Insert predictions into MongoDB
        predictions_collection.insert_one(prediction_doc)

        print(f"Predictions stored for securityId: {security_id}, period: {period}, date: {date}.")