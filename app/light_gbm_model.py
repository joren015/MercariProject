import pickle
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelBinarizer

from app.preprocessing import fit_and_save_vectorizer, split_cat
from app.train import rmsle

pd.set_option('max_colwidth', 200)


class LightGBMModel(BaseEstimator, RegressorMixin):
    def __init__(self, experiment="LightGBM", **kwarg):
        self.experiment = experiment
        self.model = LGBMRegressor(n_estimators=200,
                                   learning_rate=0.5,
                                   num_leaves=125,
                                   random_state=42)
        self.vectorizers = {}
        self.vectorizer_guid = str(uuid4()).replace('-', '_')
        self.eval_metric = make_scorer(self.score, greater_is_better=False)
        self.metrics = {
            "Root mean squared log error": self.eval_metric,
            "Explained variance": "explained_variance",
            "Max error": "max_error",
            "Negative mean absolute error": "neg_mean_absolute_error",
            "Negative mean squared error": "neg_mean_squared_error",
            "Negative root mean squared error": "neg_root_mean_squared_error",
            "Negative mean squared log error": "neg_mean_squared_log_error",
            "Negitive median absolute error": "neg_median_absolute_error",
            "R2": "r2"
        }

    def fit(self, X, y):
        self.model = LGBMRegressor(n_estimators=200,
                                   learning_rate=0.5,
                                   num_leaves=125,
                                   random_state=42)
        self.vectorizer_guid = str(uuid4()).replace('-', '_')
        self.preprocess(X)
        X = self.apply_preprocessing(X)
        y = np.log1p(y)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.apply_preprocessing(X)
        y_preds = self.model.predict(X)
        return np.expm1(y_preds)

    def score(self, y, y_pred):
        return -1 * rmsle(y, y_pred)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def preprocess(self, X):
        df = self.common_preprocessing(X)

        print("Vectorizing name")
        # Convert "name" with feature vectorization
        fit_and_save_vectorizer(
            df["name"], CountVectorizer(max_features=30000),
            "data/light_gbm/{}/name_vectorizer".format(self.vectorizer_guid))
        self.vectorizers[
            "name"] = "data/light_gbm/{}/name_vectorizer/vectorizer.pkl".format(
                self.vectorizer_guid)

        print("Vectorizing item_description")
        # Convert "item_description" with feature vectorization
        fit_and_save_vectorizer(
            df['item_description'],
            TfidfVectorizer(max_features=50000,
                            ngram_range=(1, 3),
                            stop_words='english'),
            "data/light_gbm/{}/item_description_vectorizer".format(
                self.vectorizer_guid))
        self.vectorizers[
            "item_description"] = "data/light_gbm/{}/item_description_vectorizer/vectorizer.pkl".format(
                self.vectorizer_guid)

        # Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
        # Convert each feature (cat_dae, cat_jung, cat_so) to one-hot-encoded spare matrix
        for col in [
                "brand_name", "item_condition_id", "shipping", "cat_dae",
                "cat_jung", "cat_so"
        ]:
            print("Vectorizing {}".format(col))
            save_path = "data/light_gbm/{}/{}_vectorizer".format(
                self.vectorizer_guid, col)
            fit_and_save_vectorizer(df[col],
                                    LabelBinarizer(sparse_output=True),
                                    save_path)
            self.vectorizers[col] = "{}/vectorizer.pkl".format(save_path)

    def common_preprocessing(self, X):
        # Calls split_cat() function above and create cat_dae, cat_jung, cat_so columns in X
        X['category_list'] = X['category_name'].apply(lambda x: split_cat(x))

        X['cat_dae'] = X['category_list'].apply(lambda x: x[0])
        X['cat_jung'] = X['category_list'].apply(lambda x: x[1])
        X['cat_so'] = X['category_list'].apply(lambda x: x[2])

        X.drop('category_list', axis=1, inplace=True)

        # Handling Null Values
        X['brand_name'] = X['brand_name'].fillna(value='Other_Null')
        X['category_name'] = X['category_name'].fillna(value='Other_Null')
        X['item_description'] = X['item_description'].fillna(
            value='Other_Null')
        return X

    def apply_preprocessing(self, X):
        X = self.common_preprocessing(X)

        sparse_matrix_list = []
        for col, v in self.vectorizers.items():
            with open(v, 'rb') as f:
                vectorizer = pickle.load(f)
                if col not in ["name", "item_description"]:
                    if col in ["item_condition_id", "shipping"]:
                        X.loc[~X[col].isin(vectorizer.classes_), col] = 0
                    else:
                        X.loc[~X[col].isin(vectorizer.classes_),
                              col] = "Other_Null"

                sparse_matrix_list.append(vectorizer.transform(X[col]))

        X = hstack(sparse_matrix_list).tocsr()
        return X

    def my_evaluate(self, X, y, n_splits=10, n_repeats=5, n_jobs=1):
        n_jobs = 1
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        experiment_names = [x.name for x in mlflow.list_experiments()]
        if self.experiment not in experiment_names:
            mlflow.create_experiment(self.experiment)

        experiment_id = mlflow.get_experiment_by_name(
            self.experiment).experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as run:
            cv = RepeatedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=42)
            results = cross_validate(self,
                                     X,
                                     y,
                                     scoring=self.metrics,
                                     cv=cv,
                                     n_jobs=n_jobs,
                                     return_estimator=True,
                                     verbose=1)

            results["fit_time"] = [np.double(x) for x in results["fit_time"]]
            for k, v in results.items():
                for i in range(len(v)):
                    if k == "estimator":
                        mlflow.log_params(v[i].get_params(deep=True))
                    else:
                        mlflow.log_metric(k, v[i])
