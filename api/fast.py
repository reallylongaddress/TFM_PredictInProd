from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pandas import DataFrame
from datetime import datetime
import pytz
import joblib

app = FastAPI()
# from .factory import create_app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"greeting": "These are not the droids you're looking for"}

@app.get('/predict')
def predict(pu_dt, pu_lon, pu_lat, do_lon, do_lat, pc):

    user_timezone = pytz.timezone("US/Eastern")
    pickup_naive = datetime.strptime(pu_dt, "%Y-%m-%d %H:%M:%S")
    user_pickup_dt = user_timezone.localize(pickup_naive, is_dst=None)
    pickup_datetime_utc = user_pickup_dt.astimezone(pytz.utc)
    formatted_pickup_datetime_utc = pickup_datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    dummy_id = '1999-12-31 17:18:00'
    X_pred = DataFrame(
        data=[[dummy_id, formatted_pickup_datetime_utc,
               pu_lon, pu_lat, do_lon, do_lat, pc]],
        columns = ['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
        )
    # print(f'A=====X_pred.columns: {X_pred.columns}')
    # print(f'A=====X_pred.dtypes: {X_pred.dtypes}')

    # convert strings/objects to floats
    X_pred_float = X_pred.astype({'pickup_longitude':'float64',
                                  'pickup_latitude':'float64',
                                  'dropoff_longitude':'float64',
                                  'dropoff_latitude':'float64'})

    # print(f'2=====X_pred_float.columns: {X_pred_float.columns}')
    # print(f'2=====X_pred_float_pred.dtypes: {X_pred_float.dtypes}')

    model = joblib.load('./model.joblib')
    print('====Have model')
    prediction = round(model.predict(X_pred_float)[0],2)
    return {'prediction': prediction}

'''
http://127.0.0.1:8001/predict?pu_dt=2013-07-06%2017:18:00&pu_lon=-73.950655&pu_lat=40.783282&do_lon=-73.984365&do_lat=40.269802&pc=1
'''
