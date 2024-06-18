#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import pickle
import pandas as pd






# In[3]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[11]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet')


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[16]:


print(y_pred.mean())




import requests




app = Flask('duration-prediction')


@app.route('/starter', methods=['POST'])
def predict_endpoint():
    # ride = request.get_json()
    # year = ride['year']
    # month = ride['month']

    df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet')


    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)



    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration_mean': pred.mean()
    }

    return jsonify(result)


if __name__ == "__main__":
    ride = {
    "year": 2023,
    "month": 5,

    }
    import requests

    url = 'http://localhost:9696/starter'
    response = requests.post(url, json=ride)
    print(response.json())
    app.run(debug=True, host='0.0.0.0', port=9696)

