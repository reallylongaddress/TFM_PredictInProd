import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
from scipy.sparse import csr_matrix
#from TaxiFareModel import data

class NumericOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        # X_optimized = self.optimize_numierics(pd.DataFrame(X_.todense()))
        X_optimized = self.optimize_numierics(pd.DataFrame(X_))
        X_optimized = csr_matrix(X_optimized.values)
        return X_optimized

    def optimize_numierics(self, df):

        # in_size = df.memory_usage(index=True).sum()
        # Optimized size here
        for type in ["float", "integer"]:
            l_cols = list(df.select_dtypes(include=type))
            for col in l_cols:
                df[col] = pd.to_numeric(df[col], downcast=type)
                if type == "float":
                    df[col] = pd.to_numeric(df[col], downcast="integer")
        # out_size = df.memory_usage(index=True).sum()
        # ratio = (1 - round(out_size / in_size, 2)) * 100
        # GB = out_size / 1000000000
        # if verbose:
        #     print("RAM Reduced by {} % | {} GB".format(ratio, GB))
        return df

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extract the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'
    """

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Compute the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon
        )
        return X_[['distance']]

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(X)
        X_ = X.copy()
        # X_optimized = self.feature_engineering(pd.DataFrame(X_.todense()))
        # df_ = pd.DataFrame(X_.todense())
        df_ = pd.DataFrame(X_)
        X_optimized = self.feature_engineering(df_)
        X_optimized = csr_matrix(X_optimized.values)
        return X_optimized

    def feature_engineering(self, df):

        airport_radius = 2

        print(f'feature_engineering: {df.columns}')
        # manhattan distance <=> minkowski_distance(x1, x2, y1, y2, 1)
        df['manhattan_dist'] = self.minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                    df['pickup_longitude'], df['dropoff_longitude'], 1)
        # euclidian distance <=> minkowski_distance(x1, x2, y1, y2, 2)
        df['euclidian_dist'] = self.minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                    df['pickup_longitude'], df['dropoff_longitude'], 2)

        df['delta_lon'] = df.pickup_longitude - df.dropoff_longitude
        df['delta_lat'] = df.pickup_latitude - df.dropoff_latitude
        df['direction'] = self.calculate_direction(df.delta_lon, df.delta_lat)

        #how are are pickup/dropoff from jfk airport?
        jfk_center = (40.6441666667, -73.7822222222)

        df["jfk_lat"], df["jfk_lng"] = jfk_center[0], jfk_center[1]
        print(f'feature_engineering: {df.columns}')
        print(f'feature_engineering: {df.head()}')

        args_pickup =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                            end_lat="pickup_latitude", end_lon="pickup_longitude")
        args_dropoff =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")

        # df['pickup_distance_to_jfk'] = self.haversine_distance(df, **args_pickup)
        # df['dropoff_distance_to_jfk'] = self.haversine_distance(df, **args_dropoff)
        # df['pickup_distance_to_jfk'] = self.haversine_distance(df, start_lat='jfk_lat', start_lon='jfk_lng', end_lat='pickup_latitude', end_lon='pickup_longitude')
        df['pickup_distance_to_jfk'] = self.haversine_distance(df, 'jfk_lat', 'jfk_lng', 'pickup_latitude', 'pickup_longitude')
        df['dropoff_distance_to_jfk'] = self.haversine_distance(df, 'jfk_lat', 'jfk_lng', 'dropoff_latitude', 'dropoff_longitude')
        # df['dropoff_distance_to_jfk'] = self.haversine_distance(df, **args_dropoff)

        #how are are pickup/dropoff from lga airport?
        lga_center = (40.776927, -73.873966)

        df["lga_lat"], df["lga_lng"] = lga_center[0], lga_center[1]

        args_pickup =  dict(start_lat="lga_lat", start_lon="lga_lng",
                            end_lat="pickup_latitude", end_lon="pickup_longitude")
        args_dropoff =  dict(start_lat="lga_lat", start_lon="lga_lng",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")

        # jfk = (-73.7822222222, 40.6441666667)
        # df['pickup_distance_to_lga'] = self.haversine_distance(df, **args_pickup)
        # df['dropoff_distance_to_lga'] = self.haversine_distance(df, **args_dropoff)
        df['pickup_distance_to_lga'] = self.haversine_distance(df, 'lga_lat', 'lga_lng', 'pickup_latitude', 'pickup_longitude')
        df['dropoff_distance_to_lga'] = self.haversine_distance(df, 'lga_lat', 'lga_lng', 'dropoff_latitude', 'dropoff_longitude')

        #which pickups/dropoffs can be considered airport runs?
        df['is_airport'] = df.apply(lambda row: self.fe_is_airport(row, airport_radius), axis=1)

        # $5 bucket size, more $ higher score
    #    df['fb'] = [floor(num/5)+1 for num in df['fare_amount']]

        #drop temporary and/or useless columns columns
        # df.drop(columns=['jfk_lat', 'jfk_lng', 'lga_lat', 'lga_lng',
        #                 'pickup_distance_to_jfk', 'dropoff_distance_to_jfk',
        #                 'pickup_distance_to_lga', 'dropoff_distance_to_lga',
        #                 'delta_lon', 'delta_lat'], inplace=True)

        print(f'RETURN:  {df.columns}')
        print(f'RETURN:  {df.head()}')
        return df

    def minkowski_distance_gps(self, lat1, lat2, lon1, lon2, p):
        lat1, lat2, lon1, lon2 = [self.deg2rad(coordinate) for coordinate in [lat1, lat2, lon1, lon2]]
        y1, y2, x1, x2 = [self.rad2dist(angle) for angle in [lat1, lat2, lon1, lon2]]
        x1, x2 = [self.lng_dist_corrected(elt['x'], elt['lat']) for elt in [{'x': x1, 'lat': lat1}, {'x': x2, 'lat': lat2}]]
        return self.minkowski_distance(x1, x2, y1, y2, p)

    def minkowski_distance(self, x1, x2, y1, y2, p):
        delta_x = x1 - x2
        delta_y = y1 - y2
        return ((abs(delta_x) ** p) + (abs(delta_y)) ** p) ** (1 / p)

    def calculate_direction(self, d_lon, d_lat):
        result = np.zeros(len(d_lon))
        l = np.sqrt(d_lon**2 + d_lat**2)
        result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
        idx = (d_lon<0) & (d_lat>0)
        result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
        idx = (d_lon<0) & (d_lat<0)
        result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
        return result

    def deg2rad(self, coordinate):
        return coordinate * np.pi / 180

    def rad2dist(self, coordinate):
        earth_radius = 6371 # km
        return earth_radius * coordinate

    def lng_dist_corrected(self, lng_dist, lat):
        return lng_dist * np.cos(lat)

    def fe_is_airport(self, row, airport_radius):
        print(f'is_airport: {row}')
        if row['pickup_distance_to_lga']<airport_radius or \
        row['dropoff_distance_to_lga']<airport_radius or \
        row['pickup_distance_to_jfk']<airport_radius or \
        row['dropoff_distance_to_jfk']<airport_radius :
            return 1
        return 0

    def haversine_distance(self, df, start_lat, start_lon, end_lat, end_lon):
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

    # def haversine_distance(df,
    #                    start_lat="start_lat",
    #                    start_lon="start_lon",
    #                    end_lat="end_lat",
    #                    end_lon="end_lon"):
    #     """
    #     Calculate the great circle distance between two points
    #     on the earth (specified in decimal degrees).
    #     Vectorized version of the haversine distance for pandas df
    #     Computes distance in kms
    #     """

    #     lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(df[start_lon].astype(float))
    #     lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(df[end_lon].astype(float))
    #     dlon = lon_2_rad - lon_1_rad
    #     dlat = lat_2_rad - lat_1_rad

    #     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    #     c = 2 * np.arcsin(np.sqrt(a))
    #     haversine_distance = 6371 * c
    #     return haversine_distance
