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

    model = joblib.load('./model.joblib')
    prediction = round(model.predict(X_pred)[0],2)
    return {'prediction': prediction}
