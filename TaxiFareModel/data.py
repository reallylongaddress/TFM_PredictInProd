import pandas as pd
import numpy as np
from math import floor
from TaxiFareModel.utils import simple_time_tracker
from google.cloud import storage
from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH


@simple_time_tracker
def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    print(f'get_data_from_gcp: {path}')
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_data(df, test=False):
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

def feature_engineering(df):

    airport_radius = 2

    # manhattan distance <=> minkowski_distance(x1, x2, y1, y2, 1)
    df['manhattan_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                  df['pickup_longitude'], df['dropoff_longitude'], 1)
    # euclidian distance <=> minkowski_distance(x1, x2, y1, y2, 2)
    df['euclidian_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                  df['pickup_longitude'], df['dropoff_longitude'], 2)

    df['delta_lon'] = df.pickup_longitude - df.dropoff_longitude
    df['delta_lat'] = df.pickup_latitude - df.dropoff_latitude
    df['direction'] = calculate_direction(df.delta_lon, df.delta_lat)

    #how are are pickup/dropoff from jfk airport?
    jfk_center = (40.6441666667, -73.7822222222)

    df["jfk_lat"], df["jfk_lng"] = jfk_center[0], jfk_center[1]

    args_pickup =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                        end_lat="pickup_latitude", end_lon="pickup_longitude")
    args_dropoff =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                         end_lat="dropoff_latitude", end_lon="dropoff_longitude")

    df['pickup_distance_to_jfk'] = haversine_distance(df, **args_pickup)
    df['dropoff_distance_to_jfk'] = haversine_distance(df, **args_dropoff)

    #how are are pickup/dropoff from lga airport?
    lga_center = (40.776927, -73.873966)

    df["lga_lat"], df["lga_lng"] = lga_center[0], lga_center[1]

    args_pickup =  dict(start_lat="lga_lat", start_lon="lga_lng",
                        end_lat="pickup_latitude", end_lon="pickup_longitude")
    args_dropoff =  dict(start_lat="lga_lat", start_lon="lga_lng",
                         end_lat="dropoff_latitude", end_lon="dropoff_longitude")

    # jfk = (-73.7822222222, 40.6441666667)
    df['pickup_distance_to_lga'] = haversine_distance(df, **args_pickup)
    df['dropoff_distance_to_lga'] = haversine_distance(df, **args_dropoff)

    #which pickups/dropoffs can be considered airport runs?
    df['is_airport'] = df.apply(lambda row: fe_is_airport(row, airport_radius), axis=1)

    # $5 bucket size, more $ higher score
    df['fb'] = [floor(num/5)+1 for num in df['fare_amount']]

    #drop temporary and/or useless columns columns
    df.drop(columns=['jfk_lat', 'jfk_lng', 'lga_lat', 'lga_lng',
                     'pickup_distance_to_jfk', 'dropoff_distance_to_jfk',
                     'pickup_distance_to_lga', 'dropoff_distance_to_lga',
                     'delta_lon', 'delta_lat'], inplace=True)

    return df

def minkowski_distance_gps(lat1, lat2, lon1, lon2, p):
    lat1, lat2, lon1, lon2 = [deg2rad(coordinate) for coordinate in [lat1, lat2, lon1, lon2]]
    y1, y2, x1, x2 = [rad2dist(angle) for angle in [lat1, lat2, lon1, lon2]]
    x1, x2 = [lng_dist_corrected(elt['x'], elt['lat']) for elt in [{'x': x1, 'lat': lat1}, {'x': x2, 'lat': lat2}]]
    return minkowski_distance(x1, x2, y1, y2, p)

def minkowski_distance(x1, x2, y1, y2, p):
    delta_x = x1 - x2
    delta_y = y1 - y2
    return ((abs(delta_x) ** p) + (abs(delta_y)) ** p) ** (1 / p)

def calculate_direction(d_lon, d_lat):
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result

def deg2rad(coordinate):
    return coordinate * np.pi / 180

def rad2dist(coordinate):
    earth_radius = 6371 # km
    return earth_radius * coordinate

def lng_dist_corrected(lng_dist, lat):
    return lng_dist * np.cos(lat)

def fe_is_airport(row, airport_radius):
    if row['pickup_distance_to_lga']<airport_radius or \
       row['dropoff_distance_to_lga']<airport_radius or \
       row['pickup_distance_to_jfk']<airport_radius or \
       row['dropoff_distance_to_jfk']<airport_radius :
        return 1
    return 0

def haversine_distance(df,
                       start_lat="start_lat",
                       start_lon="start_lon",
                       end_lat="end_lat",
                       end_lon="end_lon"):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version of the haversine distance for pandas df
    Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    haversine_distance = 6371 * c
    return haversine_distance

if __name__ == '__main__':
    df = get_data_from_gcp()
