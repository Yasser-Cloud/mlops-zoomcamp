#!/usr/bin/env python
# coding: utf-8

# # In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# # In[2]:


# get_ipython().system('python -V')


# # In[4]:


import pickle
import pandas as pd


# In[14]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


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


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[16]:


print(y_pred.mean())


# In[19]:





# In[17]:


df['ride_id'] = f'{2023:04d}/{4:02d}_' + df.index.astype('str')


# In[ ]:




