import pandas as pd
import numpy as np
# from math import floor
from TaxiFareModel.utils import simple_time_tracker
from google.cloud import storage
from TaxiFareModel import params#BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, BUCKET_PRED_DATA_PATH
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, NumericOptimizer, FeatureEngineering
from sklearn.compose import ColumnTransformer
import io
# from sklearn.neighbors import KNeighborsRegressor

@simple_time_tracker
def get_train_val_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    path = f"gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path, nrows=nrows)
    # print(f'get_train_val_data_from_gcp: {nrows}/{df.shape}')
    return df

def get_pred_data_from_gcp():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{params.BUCKET_NAME}/{params.BUCKET_PRED_DATA_PATH}"
    df = pd.read_csv(path)
    return df

def get_preprocessing_pipeline():
    """defines the pipeline as a class attribute"""
    dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
    time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        # ('ohe', OneHotEncoder(handle_unknown='ignore'))
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().
        #does this only act on the columns changed by the previous stage?
    ])
    preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, [
            "pickup_latitude",
            "pickup_longitude",
            'dropoff_latitude',
            'dropoff_longitude'
        ]),
        ('time', time_pipe, ['pickup_datetime']),
        ('feature_engineering', FeatureEngineering(), ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']),
        # ('ohe', OneHotEncoder(handle_unknown='ignore'), ['dow', 'hour', 'month', 'year']),
        # ValueError: 'dow' is not in list
    ], remainder="drop")

    preprocessing_pipeline = Pipeline([
        ('preproc', preproc_pipe),
        # ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ('numeric_optimizer', NumericOptimizer()),
    ])

    return preprocessing_pipeline

def get_full_pipeline(estimator):
    pipeline = get_preprocessing_pipeline()
    pipeline.steps.append(['estimator', estimator])
    return pipeline

def clean_data(df):
    unused_column = "Unnamed: 0"
    if unused_column in df.keys():
        df = df.drop(axis=1, columns=["Unnamed: 0"])
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def save_submission(y_pred, y_keys, estimator_name, process_start_time):

    y_keys = pd.DataFrame(y_keys)
    y_keys = y_keys.reset_index(drop=True)
    y_pred_submission = pd.concat([y_keys, pd.Series(y_pred)],axis=1)
    y_pred_submission.columns = ['key', 'fare_amount']

    client = storage.Client()

    file_name = f'submission_{estimator_name}_{process_start_time}.csv'
    local_file_path = params.LOCAL_STORAGE_LOCATION + file_name

    #save locally
    pd.DataFrame(y_pred_submission).to_csv(local_file_path, index=False)

    #save go GCP, no need for local file save even though one occurs above
    f = io.StringIO()
    y_pred_submission.to_csv(f)
    f.seek(0)
    client.get_bucket(params.BUCKET_NAME).blob(params.GCM_STORAGE_LOCATION + file_name).upload_from_file(f, content_type='text/csv')

def save_model(model, estimator_name, process_start_time):
    file_name = f'gcp_model_{estimator_name}_{process_start_time}.joblib'
    local_file_path = params.LOCAL_STORAGE_LOCATION + file_name
    # print(f'model local_file_path: {local_file_path}')

    joblib.dump(model, local_file_path)

    client = storage.Client()
    bucket = client.bucket(params.BUCKET_NAME)
    blob = bucket.blob(params.GCM_STORAGE_LOCATION + file_name)

    blob.upload_from_filename(params.LOCAL_STORAGE_LOCATION + file_name)
    print(f"uploaded {params.LOCAL_STORAGE_LOCATION}{file_name} => {params.GCM_STORAGE_LOCATION}{file_name}")


if __name__ == '__main__':
    df = get_train_val_data_from_gcp(100)
    pipeline = get_preprocessing_pipeline()
