# import joblib
from termcolor import colored
import mlflow
from time import time
from TaxiFareModel import data
from TaxiFareModel import params
# from TaxiFareModel.data import get_train_val_data_from_gcp, clean_data, feature_engineering, get_preprocessing_pipeline
# from TaxiFareModel.gcp import storage_upload
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.params import MLFLOW_URI, EXPERIMENT_NAME
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

# import sys
import os
import platform

class Trainer(object):
    def __init__(self, X_train, X_val, y_train, y_val, X_pred, X_pred_keys):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.run_start_time = round(time(), 0)

        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_pred = X_pred
        self.X_pred_keys = X_pred_keys

        # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_pred.shape)
        # print(type(X_train), type(y_train), type(X_val), type(y_val), type(X_pred))

        # for MLFlow
        # self.experiment_name = data.EXPERIMENT_NAME

    # def set_experiment_name(self, experiment_name):
    #     '''defines the experiment name for MLFlow'''
    #     self.experiment_name = experiment_name

    def run(self):

        self.mlflow_experiment_id = self.get_mlflow_experiment_id()

        params = {
            'estimators': {
                'knn':{
                    'hyperparams':{
                        'n_neighbors':[25,50,100],
                        'n_jobs':[-1],
                    },
                },
                'linear':{
                    'hyperparams':{
                        'n_jobs':[-1],
                    }
                },
                # 'sgd':{
                #     'hyperparams':{
                #         'learning_rate': ['invscaling'],
                #     }
                # }
            },
            # 'nrows':1_000,
            #'nrows':200_000,
            # 'starttime':starttime,
            # 'experiment_name':params.EXPERIMENT_NAME
        }
        for estimator_name, hyperparams in params.get('estimators').items():

            loop_start_time = round(time(), 0)
            print(f'key: {estimator_name}')

            ml_flow_client = MlflowClient()
            ml_flow_run = ml_flow_client.create_run(self.mlflow_experiment_id)

            ml_flow_client.log_param(ml_flow_run.info.run_id, 'model', estimator_name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'train_size', f'{self.X_train.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'validate_size', f'{self.X_val.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'predict_size', f'{self.X_pred.shape[0]}')

            ml_flow_client.log_param(ml_flow_run.info.run_id, 'os.name', os.name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.system', platform.system())
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.release', platform.release())

            #dbd todo - this is ugly, reduce to 1 line
            for param_key, param_value in hyperparams.items():
                ml_flow_client.log_param(ml_flow_run.info.run_id, param_key, param_value)

                grid = None
                model = None
                if estimator_name == 'knn':
                    model = KNeighborsRegressor()

                elif estimator_name == 'sgd':
                    model = SGDRegressor()

                elif estimator_name == 'linear':
                    model = LinearRegression()
                else:
                    raise Exception("Unknown model type")

                grid = GridSearchCV(model,
                                    param_grid=hyperparams.get("hyperparams"),
                                    cv=3,
                                    scoring='neg_root_mean_squared_error',
                                    # return_train_score=True,
                                    verbose=1,
                                    n_jobs=-1
                                    )

                grid.fit(self.X_train, self.y_train)

                ml_flow_client.log_param(ml_flow_run.info.run_id, 'best_params', grid.best_params_)

                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'train_rmse', f'{-grid.best_score_}')

                best_model = grid.best_estimator_

                validate_rmse = self.evaluate(best_model)
                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'validate_rmse', f'{validate_rmse}')

                y_pred = best_model.predict(self.X_pred)

                data.save_submission(y_pred, self.X_pred_keys, estimator_name, self.run_start_time)
                data.save_model(best_model, estimator_name, self.run_start_time)

                loop_end_time = time()
                loop_elapsed_time = round((loop_end_time - loop_start_time), 1)
                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'elapsed_time', f'{loop_elapsed_time}')

    def evaluate(self, model):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_val_pred = model.predict(self.X_val)
        rmse = compute_rmse(y_val_pred, self.y_val)
        return round(rmse, 2)

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    # @memoized_property
    def get_mlflow_experiment_id(self):
        try:
            A = self.mlflow_client.create_experiment(params.EXPERIMENT_NAME)
            return A
        except BaseException:
            B = self.mlflow_client.get_experiment_by_name(params.EXPERIMENT_NAME).experiment_id
            return B

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # Get and clean data
    N = 1_000
    df = data.get_train_val_data_from_gcp(nrows=N)
    X_pred = data.get_pred_data_from_gcp()

    X = data.clean_data(df)
    X_pred = data.clean_data(X_pred)
    X_pred_keys = X_pred["key"]

    # df = data.feature_engineering(df)
    # X_pred = data.feature_engineering(X_pred)

    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]


    print(f'X: {type(X)}')
    print(f'X: {X.shape}')
    print(f'X: {X.columns}')
    print(f'X_pred: {type(X_pred)}')
    print(f'X_pred: {X_pred.shape}')
    print(f'X_pred: {X_pred.columns}')
    preprocessing_pipeline = data.get_preprocessing_pipeline()
    print(f'preprocessing_pipeline: {type(preprocessing_pipeline)}')
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X)
    X_pred_preprocessed = preprocessing_pipeline.transform(X_pred)

    X_train, X_test, y_train, y_test = train_test_split(X_train_preprocessed, y, test_size=0.3)

    trainer = Trainer(X_train, X_test, y_train, y_test, X_pred_preprocessed, X_pred_keys)
    trainer.run()
