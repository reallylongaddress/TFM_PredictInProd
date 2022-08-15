'''
TODO

X-Submit build to kaggle

uncomment
# ('ohe', OneHotEncoder(handle_unknown='ignore'))

X-Submit build to kaggle

Build Docker image

Run docker image locally
Publish docker image
--hit docker image from vs code
--hit docker image from notebook


PUsh docker image to GCP
Publish docker image
--hit docker image from vs code
--hit docker image from notebook

build web FE
push web FE to Heroku

Improve projections
--export DF and optimize
--Try Neural Network

'''


import os
import platform
from time import time
# import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
# from termcolor import colored

from TaxiFareModel import data
from TaxiFareModel import params
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.params import MLFLOW_URI

from memoized_property import memoized_property

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet

class Trainer2(object):
    def __init__(self, num_train_rows=1_000):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.run_start_time = round(time(), 0)
        self.mlflow_experiment_id = self.get_mlflow_experiment_id()
        self.num_train_rows = num_train_rows

    def run(self):

        df = data.get_train_val_data_from_gcp(nrows=self.num_train_rows)
        X_pred = data.get_pred_data_from_gcp()

        X = data.clean_data(df)

        pred_data_keys = X_pred["key"].copy()

        X = df.drop("fare_amount", axis=1)
        y = df["fare_amount"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        #dbd undo this hack
        self.X_test = X_test
        self.y_test = y_test

        params = {
            'estimators': {
                'knn':{
                    'hyperparams':{
                        'estimator__n_neighbors':[10,25, 50],
                    },
                },
                # 'linear':{
                #     'hyperparams':{
                #         'estimator__n_jobs':[-1],
                #     }
                # },
                'lasso':{
                    'hyperparams':{
                        'estimator__alpha':[1],
                        'estimator__max_iter':[1000,5000,10000],
                        'estimator__tol':[1e-4,1e-2,1],
                        'estimator__selection':['cyclic','random'],
                    }
                },
                'elasticnet':{
                    'hyperparams':{
                        'estimator__alpha':[1],
                        'estimator__l1_ratio':[.5],
                        'estimator__max_iter':[1000,5000,10000],
                        'estimator__tol':[1e-4,1e-2,1],
                        'estimator__selection':['cyclic','random'],
                    }
                },
                'ridge':{
                    'hyperparams':{
                        'estimator__max_iter':[None],
                        'estimator__solver':['auto'],
                    }
                },
                # 'sgd':{
                #     'hyperparams':{
                #         'learning_rate': ['invscaling'],
                #     }
                # }
            },
        }

        for estimator_name, hyperparams in params.get('estimators').items():

            loop_start_time = round(time(), 0)

            ml_flow_client = MlflowClient()
            ml_flow_run = ml_flow_client.create_run(self.mlflow_experiment_id)

            ml_flow_client.log_param(ml_flow_run.info.run_id, 'model', estimator_name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'train_size', f'{X_train.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'validate_size', f'{X_test.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'predict_size', f'{X_pred.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'run_start_time', f'{self.run_start_time}')

            ml_flow_client.log_param(ml_flow_run.info.run_id, 'os.name', os.name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.system', platform.system())
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.release', platform.release())

            model = None
            if estimator_name == 'knn':
                model = KNeighborsRegressor()

            elif estimator_name == 'sgd':
                model = SGDRegressor()

            elif estimator_name == 'linear':
                model = LinearRegression()

            elif estimator_name == 'lasso':
                model = Lasso()

            elif estimator_name == 'elasticnet':
                model = ElasticNet()

            elif estimator_name == 'ridge':
                model = Ridge()
            else:
                raise Exception("Unknown model type")

            print(f'estimator_name: {estimator_name}')

            pipeline = data.get_full_pipeline(model)

            for param_key, param_value in hyperparams.items():
                ml_flow_client.log_param(ml_flow_run.info.run_id, param_key, param_value)
                print(f'param_key: {param_key}, param_value: {param_value}')
                grid = GridSearchCV(pipeline,
                                    param_grid=hyperparams.get("hyperparams"),
                                    cv=2,
                                    scoring='neg_root_mean_squared_error',
                                    # return_train_score=True,
                                    verbose=1,
                                    n_jobs=-1
                                    )

                grid.fit(X_train, y_train)

                ml_flow_client.log_param(ml_flow_run.info.run_id, 'best_params', grid.best_params_)

                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'train_rmse', f'{-grid.best_score_}')

                best_model = grid.best_estimator_

                validate_rmse = self.evaluate(best_model)
                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'validate_rmse', f'{validate_rmse}')

                y_pred = best_model.predict(X_pred)

                data.save_submission(y_pred, pred_data_keys, estimator_name, self.run_start_time)
                data.save_model(best_model, estimator_name, self.run_start_time)

                loop_end_time = time()
                loop_elapsed_time = round((loop_end_time - loop_start_time), 1)
                ml_flow_client.log_metric(ml_flow_run.info.run_id, 'elapsed_time', f'{loop_elapsed_time}')

    def evaluate(self, model):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_test_pred = model.predict(self.X_test).round(decimals=2)

        rmse = compute_rmse(y_test_pred, self.y_test)
        return round(rmse, 2)

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    # @memoized_property
    def get_mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(params.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(params.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # Get and clean data
    N = 100_000

    trainer2 = Trainer2(N)
    trainer2.run()
