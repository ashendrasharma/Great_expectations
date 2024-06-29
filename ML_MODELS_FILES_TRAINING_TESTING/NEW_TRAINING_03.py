# -*- coding: utf-8 -*-
"""MODEL_TRAIN01.ipynb
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from datetime import datetime
import os

# Connect to MongoDB
client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
db = client['intelliinvest']
fundamentals_collection = db['STOCK_FUNDAMENTALS']
signals_collection = db['STOCK_SIGNALS_COMPONENTS_10']
details_collection = db['STOCK_DETAILS']

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

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Load and preprocess data
data = []

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
    df.loc[:, 'signalDate'] = pd.to_datetime(df['signalDate'])

    data.append(df)

df = pd.concat(data)

# Use predefined categories
def assign_risk_category(row):
    if row['stdDevOfReturn'] >= 0 and row['stdDevOfReturn'] <= 0.25:
        return 'low'
    elif row['stdDevOfReturn'] > 0.25 and row['stdDevOfReturn'] <= 0.45:
        return 'medium'
    else:
        return 'high'

df['category'] = df.apply(assign_risk_category, axis=1)

# Convert signalDate to float (timestamp)
df['signalDate'] = df['signalDate'].apply(lambda x: x.timestamp())

# Train models for each industry and risk category
for industry in industries:
    industry_df = df[df['industry'] == industry]

    for category in ['low', 'medium', 'high']:
        category_df = industry_df[industry_df['category'] == category]

        if len(category_df) < 3:
            print(f"Not enough data for {industry} - {category} risk, skipping.")
            continue

        X = category_df.drop(columns=['closePrice', 'industry', 'category'])
        y = category_df['closePrice']

        models = {
            'lr': LinearRegression(),
            'dt': DecisionTreeRegressor(),
            'gb': GradientBoostingRegressor()
        }

        for model_name, model in models.items():
            model.fit(X, y)
            model_filename = f'models/{model_name}_model_{industry}_{category}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Trained and saved {model_name} model for {industry} - {category} risk.")

print("Training completed and models saved.")
