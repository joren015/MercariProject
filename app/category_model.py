import pickle
import re
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (explained_variance_score, make_scorer, max_error,
                             mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from app.preprocessing import fit_and_save_vectorizer
from app.train import rmsle

pd.set_option('max_colwidth', 200)


class CategoryModel(BaseEstimator, RegressorMixin):
    def __init__(self, experiment="CategoryModel", **kwarg):
        self.experiment = experiment
        self.df_array = {}
        self.models = {}
        self.trained_categories = []
        self.vectorizer_guid = str(uuid4()).replace('-', '_')
        self.eval_metric = make_scorer(self.score, greater_is_better=False)
        self.metrics = {
            "Root mean squared log error":
            lambda y, y_pred: -1 * rmsle(y, y_pred),
            "Explained variance": explained_variance_score,
            "Max error": max_error,
            "Negative mean absolute error": mean_absolute_error,
            "Negative mean squared error": mean_squared_error,
            "Negative root mean squared error": mean_squared_error,
            "Negative mean squared log error": mean_squared_log_error,
            "Negitive median absolute error": median_absolute_error,
            "R2": r2_score
        }
        self.example_model = LGBMRegressor(n_estimators=50,
                                           learning_rate=0.5,
                                           num_leaves=125,
                                           random_state=156)

    def fit(self, X, y):
        self.models = {}
        self.vectorizer_guid = str(uuid4()).replace('-', '_')
        Xp = X.copy(deep=True)
        self.preprocess(Xp)
        del Xp
        X = self.apply_preprocessing(X)
        unique_categories = list(set(X.keys()))
        for category in unique_categories:
            if category in self.trained_categories:
                print("Fitting {}".format(category))
                df, x = X[category]["df"], X[category]["x"]
                y = df["price"]
                lgbm_model = LGBMRegressor(n_estimators=50,
                                           learning_rate=0.5,
                                           num_leaves=125,
                                           random_state=156)
                if len(y) >= 2:
                    lgbm_model = lgbm_model.fit(x, y)
                    self.models[category]["model"] = lgbm_model

    def predict(self, X):
        X["predict_id"] = np.arange(X.shape[0])
        Xp = X.copy(deep=True)
        Xp = self.apply_preprocessing(Xp)
        results = pd.DataFrame(columns=["id", "y"])
        for k, v in Xp.items():
            if k == "Handmade/Housewares/Lighting":
                print("Breakpoint")

            df = v["df"]
            if self.models[k]["model"] is not None:
                yi = self.models[k]["model"].predict(v["x"])
                result = pd.DataFrame.from_dict({
                    "id": df["predict_id"].tolist(),
                    "y": yi
                })
            else:
                print("Unable to predict for {}".format(k))
                print([0 for x in range(df.shape[0])])
                result = pd.DataFrame.from_dict({
                    "id":
                    df["predict_id"].tolist(),
                    "y": [0.0 for x in range(len(df["predict_id"].tolist()))]
                })
            results = pd.concat([results, result])

        results.sort_values(by=["id"]).reset_index(drop=True)
        return results["y"]

    def score(self, y, y_pred):
        return -1 * rmsle(y, y_pred)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def model_train_predict(self, model, matrix_list_X, matrix_list_Y, df,
                            df_test):
        X = hstack(matrix_list_X).tocsr()
        Y = hstack(matrix_list_Y).tocsr()
        X_price = df['price']
        Y_price = df_test['price']

        # 모델 학습 및 예측
        # Model traiing and test
        preds = []
        try:
            model.fit(X, X_price)
            preds = model.predict(Y)
        except:
            print("ERROR")
            preds = [0]
            Y_price = [1000000]
            np.array(preds)
            np.array(Y_price)

        return preds, Y_price, model

    def preprocess(self, X):
        X = self.common_preprocessing(X)
        df_array = self.group_by_category(X, impute=False)

        for category in df_array:
            self.trained_categories.append(category)
            self.models[category] = {"vectorizers": {}, "model": None}
            train_cat = ""
            if category in df_array:
                train_cat = category
            else:
                split = category.split('/')
                train_cat = split[0] + '/' + split[1]

                for category in df_array:
                    if category.find(train_cat) != -1:
                        train_cat = category
                        break

            print("Creating vectorizer for {}".format(train_cat))
            df = pd.DataFrame(df_array[train_cat])
            combo = df

            # Convert "name" with feature vectorization
            fit_and_save_vectorizer(
                combo.name, CountVectorizer(max_features=30000),
                "data/category_model/{}/name_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "name"] = "data/category_model/{}/name_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            # Convert "item_description" with feature vectorization
            fit_and_save_vectorizer(
                combo["item_description"],
                TfidfVectorizer(max_features=50000,
                                ngram_range=(1, 3),
                                stop_words='english'),
                "data/category_model/{}/item_description_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "item_description"] = "data/category_model/{}/item_description_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            # Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
            fit_and_save_vectorizer(
                combo["brand_name"], LabelBinarizer(sparse_output=True),
                "data/category_model/{}/brand_name_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "brand_name"] = "data/category_model/{}/brand_name_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            fit_and_save_vectorizer(
                combo["item_condition_id"], LabelBinarizer(sparse_output=True),
                "data/category_model/{}/item_condition_id_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "item_condition_id"] = "data/category_model/{}/item_condition_id_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            fit_and_save_vectorizer(
                combo["shipping"], LabelBinarizer(sparse_output=True),
                "data/category_model/{}/shipping_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "shipping"] = "data/category_model/{}/shipping_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

    def common_preprocessing(self, X):
        X['brand_name'] = X['brand_name'].fillna(value='Other_Null')
        X['category_name'] = X['category_name'].fillna(value='Other_Null')
        X['item_description'] = X['item_description'].fillna(
            value='Other_Null')
        return X

    def group_by_category(self, X, impute=False):
        if impute:
            X.loc[~X["category_name"].isin(self.models.keys()),
                  "category_name"] = "Other_Null"

        training_grouped = X.groupby(X.category_name)
        id_array = []
        for t in training_grouped.category_name:
            id_array.append(t)

        df_array = {}
        for t in id_array:
            df_array[t[0]] = (training_grouped.get_group(t[0]))

        return df_array

    def apply_preprocessing(self, X):
        X = self.common_preprocessing(X)
        df_array = self.group_by_category(X, impute=True)

        sparse_matrix_dict = {}
        for category, _ in df_array.items():
            print("Applying preprocessing {}".format(category))
            df = pd.DataFrame(df_array[category])
            sparse_matrix_list = []
            for col in [
                    "name", "item_description", "brand_name",
                    "item_condition_id", "shipping"
            ]:
                v = self.models[category]["vectorizers"][col]
                with open(v, 'rb') as f:
                    vectorizer = pickle.load(f)
                    if col not in ["name", "item_description"]:
                        if col in ["item_condition_id", "shipping"]:
                            X.loc[~X[col].isin(vectorizer.classes_), col] = 0
                        else:
                            X.loc[~X[col].isin(vectorizer.classes_),
                                  col] = "Other_Null"

                    sparse_matrix_list.append(vectorizer.transform(df[col]))

            sparse_matrix_dict[category] = {
                "df": df,
                "x": hstack(sparse_matrix_list).tocsr()
            }

        return sparse_matrix_dict

    def my_evaluate(self, X, y, n_splits=10, n_repeats=5):
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        experiment_names = [x.name for x in mlflow.list_experiments()]
        if self.experiment not in experiment_names:
            mlflow.create_experiment(self.experiment)

        experiment_id = mlflow.get_experiment_by_name(
            self.experiment).experiment_id
        i = 0
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_params(self.example_model.get_params(deep=True))
            cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                         n_repeats=n_repeats,
                                         random_state=42)
            X['category_name'] = X['category_name'].fillna(value='Other_Null')
            for train_index, test_index in cv.split(X, X["category_name"]):
                print("ITERATION: {}".format(i))
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]

                self.fit(X_train, y_train)
                total_lgbm_preds = pd.array([], dtype=float)
                total_yi = pd.array([], dtype=float)
                for category in X_test["category_name"].unique():
                    category_index = X_test.index[X_test["category_name"] ==
                                                  category]
                    Xi = X_test.loc[category_index]
                    yi = y_test.loc[category_index]
                    results = self.predict(Xi)
                    lgbm_preds = abs(results)
                    total_lgbm_preds = np.concatenate(
                        [total_lgbm_preds, lgbm_preds])
                    total_yi = np.concatenate([total_yi, yi])
                    # logged_metrics = {}
                    # for k, metric in self.metrics.items():
                    #     metric_key = re.sub(
                    #         r'[^a-zA-Z0-9]', '',
                    #         "{} split {} {}".format(category, i, k))
                    #     if k == "Test Negative root mean squared error":
                    #         value = metric(yi, lgbm_preds, squared=True)
                    #     else:
                    #         value = metric(yi, lgbm_preds)

                    #     logged_metrics[metric_key] = value

                    # mlflow.log_metrics(logged_metrics)

                logged_metrics = {}
                for k, metric in self.metrics.items():
                    metric_key = re.sub(r'[^a-zA-Z0-9]', '',
                                        "total split {} {}".format(i, k))
                    if k == "Test Negative root mean squared error":
                        value = metric(total_yi,
                                       total_lgbm_preds,
                                       squared=True)
                    else:
                        value = metric(total_yi, total_lgbm_preds)

                    logged_metrics[metric_key] = value

                mlflow.log_metrics(logged_metrics)
                i += 1
