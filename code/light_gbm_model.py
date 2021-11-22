import pickle
from os import makedirs

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer

pd.set_option('max_colwidth', 200)


def rmsle(y, y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))


# Create split_cat() function that returns cateogires (dae, jung, so) called by "apply lambda"
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null', 'Other_Null', 'Other_Null']


def fit_and_save_vectorizer(X, vectorizer, folder_name):
    makedirs(folder_name, exist_ok=True)
    X_vectorized = vectorizer.fit_transform(X)
    with open("{}/vectorizer.pkl".format(folder_name), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open("{}/X.pkl".format(folder_name), 'wb') as f:
        pickle.dump(X, f)

    with open("{}/X_vectorized.pkl".format(folder_name), 'wb') as f:
        pickle.dump(X_vectorized, f)

    return vectorizer, X_vectorized


class LightGBMModel(BaseEstimator, RegressorMixin):
    def __init__(self, experiment="LightGBM", **kwarg):
        self.experiment = experiment
        self.model = LGBMRegressor(n_estimators=200,
                                   learning_rate=0.5,
                                   num_leaves=125,
                                   random_state=42)
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
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, y, y_pred):
        return -1 * self.evaluate_org_price(y, y_pred)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def evaluate_org_price(self, y_test, preds):
        # In the original dataset, we need to convert the log1p back to its original form.
        # 원본 데이터는 log1p로 변환되었으므로 exmpm1으로 원복 필요.
        preds_exmpm = np.expm1(preds)
        y_test_exmpm = np.expm1(y_test)

        # rmsle로 RMSLE 값 추출
        rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
        return rmsle_result

    def get_dataset(self, nrows=-1):
        if nrows > 0:
            mercari_df = pd.read_csv('./data/train.tsv',
                                     sep='\t',
                                     usecols=["train_id", "price"],
                                     nrows=nrows)
        else:
            mercari_df = pd.read_csv('./data/train.tsv',
                                     sep='\t',
                                     usecols=["train_id", "price"])

        mercari_df['price'] = np.log1p(mercari_df['price'])
        X_name = pickle.load(
            open("data/name_count_vectorizer/X_vectorized.pkl", "rb"))
        X_descp = pickle.load(
            open("data/item_description_tfidf_vectorizer/X_vectorized.pkl",
                 "rb"))
        X_brand = pickle.load(
            open("data/brand_name_label_binarizer/X_vectorized.pkl", "rb"))
        X_item_cond_id = pickle.load(
            open("data/item_condition_id_label_binarizer/X_vectorized.pkl",
                 "rb"))
        X_shipping = pickle.load(
            open("data/shipping_label_binarizer/X_vectorized.pkl", "rb"))
        X_cat_dae = pickle.load(
            open("data/cat_dae_label_binarizer/X_vectorized.pkl", "rb"))
        X_cat_jung = pickle.load(
            open("data/cat_jung_label_binarizer/X_vectorized.pkl", "rb"))
        X_cat_so = pickle.load(
            open("data/cat_so_label_binarizer/X_vectorized.pkl", "rb"))
        sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id,
                              X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
        X = hstack(sparse_matrix_list).tocsr()
        y = mercari_df['price']
        return X, y
