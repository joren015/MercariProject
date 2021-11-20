print("Importing packages")
import gc

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     train_test_split)
from sklearn.preprocessing import LabelBinarizer

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
mercari_df = pd.read_csv('./data/train.tsv', sep='\t')

print("Preprocessing")
mercari_df['price'] = np.log1p(mercari_df['price'])
boolean_cond = mercari_df['item_description'] == 'No description yet'

# Calls split_cat() function above and create cat_dae, cat_jung, cat_so columns in mercari_df
mercari_df['category_list'] = mercari_df['category_name'].apply(
    lambda x: split_cat(x))
mercari_df['category_list'].head()

mercari_df['cat_dae'] = mercari_df['category_list'].apply(lambda x: x[0])
mercari_df['cat_jung'] = mercari_df['category_list'].apply(lambda x: x[1])
mercari_df['cat_so'] = mercari_df['category_list'].apply(lambda x: x[2])

mercari_df.drop('category_list', axis=1, inplace=True)

# Handling Null Values
mercari_df['brand_name'] = mercari_df['brand_name'].fillna(value='Other_Null')
mercari_df['category_name'] = mercari_df['category_name'].fillna(
    value='Other_Null')
mercari_df['item_description'] = mercari_df['item_description'].fillna(
    value='Other_Null')

gc.collect()

# Convert "name" with feature vectorization
cnt_vec = CountVectorizer(max_features=30000)
X_name = cnt_vec.fit_transform(mercari_df.name)

# Convert "item_description" with feature vectorization
tfidf_descp = TfidfVectorizer(max_features=50000,
                              ngram_range=(1, 3),
                              stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

print("tfidf_descp type: {}".format(type(tfidf_descp)))
print("X_descp type: {}".format(type(X_descp)))

# Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
lb_brand_name = LabelBinarizer(sparse_output=True)
X_brand = lb_brand_name.fit_transform(mercari_df['brand_name'])

print("lb_brand_name type: {}".format(type(lb_brand_name)))
print("X_brand type: {}".format(type(X_brand)))

lb_item_cond_id = LabelBinarizer(sparse_output=True)
X_item_cond_id = lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])

lb_shipping = LabelBinarizer(sparse_output=True)
X_shipping = lb_shipping.fit_transform(mercari_df['shipping'])

# Convert each feature (cat_dae, cat_jung, cat_so) to one-hot-encoded spare matrix
lb_cat_dae = LabelBinarizer(sparse_output=True)
X_cat_dae = lb_cat_dae.fit_transform(mercari_df['cat_dae'])

lb_cat_jung = LabelBinarizer(sparse_output=True)
X_cat_jung = lb_cat_jung.fit_transform(mercari_df['cat_jung'])

lb_cat_so = LabelBinarizer(sparse_output=True)
X_cat_so = lb_cat_so.fit_transform(mercari_df['cat_so'])

gc.collect()

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping,
                      X_cat_dae, X_cat_jung, X_cat_so)
X_features_sparse = hstack(sparse_matrix_list).tocsr()

del X_features_sparse
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

from lightgbm import LGBMRegressor

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
