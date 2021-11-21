print("Importing packages")
import gc
import pickle

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     train_test_split)

pd.set_option('max_colwidth', 200)


# Create split_cat() function that returns cateogires (dae, jung, so) called by "apply lambda"
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null', 'Other_Null', 'Other_Null']


def rmsle(y, y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))


def evaluate_org_price(y_test, preds):
    # In the original dataset, we need to convert the log1p back to its original form.
    # 원본 데이터는 log1p로 변환되었으므로 exmpm1으로 원복 필요.
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)

    # rmsle로 RMSLE 값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result


def model_train_predict(model, matrix_list):
    # scipy.sparse 모듈의 hstack 을 이용하여 sparse matrix 결합
    X = hstack(matrix_list).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        mercari_df['price'],
                                                        test_size=0.2,
                                                        random_state=156)

    # 모델 학습 및 예측
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    del X, X_train, X_test, y_train
    gc.collect()

    return preds, y_test


def rmsle_eval(y_test, y_pred):
    return -1 * evaluate_org_price(y_test, y_pred)


print("Loading dataset")
mercari_df = pd.read_csv('./data/train.tsv',
                         sep='\t',
                         usecols=["train_id", "price"])
mercari_df['price'] = np.log1p(mercari_df['price'])
X_name = pickle.load(open("data/name_count_vectorizer/X_vectorized.pkl", "rb"))
X_descp = pickle.load(
    open("data/item_description_tfidf_vectorizer/X_vectorized.pkl", "rb"))
X_brand = pickle.load(
    open("data/brand_name_label_binarizer/X_vectorized.pkl", "rb"))
X_item_cond_id = pickle.load(
    open("data/item_condition_id_label_binarizer/X_vectorized.pkl", "rb"))
X_shipping = pickle.load(
    open("data/shipping_label_binarizer/X_vectorized.pkl", "rb"))
X_cat_dae = pickle.load(
    open("data/cat_dae_label_binarizer/X_vectorized.pkl", "rb"))
X_cat_jung = pickle.load(
    open("data/cat_jung_label_binarizer/X_vectorized.pkl", "rb"))
X_cat_so = pickle.load(
    open("data/cat_so_label_binarizer/X_vectorized.pkl", "rb"))

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping,
                      X_cat_dae, X_cat_jung, X_cat_so)
# X_features_sparse = hstack(sparse_matrix_list).tocsr()

# del X_features_sparse
gc.collect()

linear_model = Ridge(solver="lsqr", fit_intercept=False)

# RMSLE without Item Description
sparse_matrix_list = (X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae,
                      X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model,
                                           matrix_list=sparse_matrix_list)
print('Item Description을 제외했을 때 rmsle 값:',
      evaluate_org_price(y_test, linear_preds))

# RMSLE with Item Description
sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping,
                      X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model,
                                           matrix_list=sparse_matrix_list)
print('Item Description을 포함한 rmsle 값:',
      evaluate_org_price(y_test, linear_preds))

gc.collect()

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping,
                      X_cat_dae, X_cat_jung, X_cat_so)

# RMSLE of LGBM
lgbm_model = LGBMRegressor(n_estimators=200,
                           learning_rate=0.5,
                           num_leaves=125,
                           random_state=156)
lgbm_preds, y_test = model_train_predict(model=lgbm_model,
                                         matrix_list=sparse_matrix_list)
print('LightGBM rmsle 값:', evaluate_org_price(y_test, lgbm_preds))

preds = lgbm_preds * 0.45 + linear_preds * 0.55
print('LightGBM과 Ridge를 ensemble한 최종 rmsle 값:',
      evaluate_org_price(y_test, preds))

mlflow.sklearn.autolog()
with mlflow.start_run() as run:
    rmsle_metric = make_scorer(rmsle_eval, greater_is_better=False)
    metrics = {
        "Root mean squared log error": rmsle_metric,
        "Explained variance": "explained_variance",
        "Max error": "max_error",
        "Negative mean absolute error": "neg_mean_absolute_error",
        "Negative mean squared error": "neg_mean_squared_error",
        "Negative root mean squared error": "neg_root_mean_squared_error",
        "Negative mean squared log error": "neg_mean_squared_log_error",
        "Negitive median absolute error": "neg_median_absolute_error",
        "R2": "r2"
    }
    param_grid = {"alpha": [0.01, 0.1, 1]}

    linear_model = Ridge(solver="lsqr", fit_intercept=False)

    sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping,
                          X_cat_dae, X_cat_jung, X_cat_so)
    X = hstack(sparse_matrix_list).tocsr()
    y = mercari_df['price']
    cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
    gs = GridSearchCV(linear_model,
                      scoring=metrics,
                      refit="Root mean squared log error",
                      param_grid=param_grid,
                      cv=cv,
                      n_jobs=1,
                      return_train_score=True,
                      verbose=4)
    gs.fit(X, y)
    results = gs.cv_results_
