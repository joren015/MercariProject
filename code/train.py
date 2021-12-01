import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from model_logger import run

pd.set_option('max_colwidth', 200)
model_type = "NN"


def rmsle(y, y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))


# import gc

# import scipy.sparse as sp
# from scipy.sparse import hstack

# def model_train_predict(model, matrix_list_X, matrix_list_Y, df, df_test):
#     # scipy.sparse 모듈의 hstack 을 이용하여 sparse matrix 결합
#     #
#     X = hstack(matrix_list_X).tocsr()
#     Y = hstack(matrix_list_Y).tocsr()
#     # X = matrix_list_X
#     # Y = matrix_list_Y

#     # X_train, X_test, y_train, y_test=train_test_split(X, df['price'],
#     #                                                   test_size=0.1)

#     # Y_train, Y_test, y_price, y_useless=train_test_split(Y, df_test['price'],
#     #                                                   test_size=0.1)

#     X_price = df['price']
#     Y_price = df_test['price']

#     # 모델 학습 및 예측
#     # Model traiing and test
#     preds = []
#     try:
#         model.fit(X, X_price)
#         preds = model.predict(Y)
#     except:
#         print("ERROR")
#         preds = [0]
#         Y_price = [1000000]
#         np.array(preds)
#         np.array(Y_price)

#     #del X , X_train , X_test , Y
#     gc.collect()

#     return preds, Y_price

if __name__ == "__main__":
    mercari_df = pd.read_csv('data/train.tsv', sep='\t', nrows=10000)
    X = mercari_df
    y = mercari_df["price"]

    print("Preprocessing dataset")
    if model_type == "light_gbm":
        from light_gbm_model import LightGBMModel
        model = LightGBMModel()
        run(model, X, y, model.metrics, experiment=model.experiment, n_jobs=1)
    elif model_type == "NN":
        from nn_model import NNModel
        model = NNModel()
        model.my_evaluate(X, y)
    else:
        from category_model import CategoryModel
        model = CategoryModel()
        model.fit(mercari_df)

        # df_train = df.sample(frac=0.8, random_state=25)
        # df_test = df.drop(df_train.index)

        # df_train['brand_name'] = df_train['brand_name'].fillna(
        #     value='Other_Null')
        # df_train['category_name'] = df_train['category_name'].fillna(
        #     value='Other_Null')
        # df_train['item_description'] = df_train['item_description'].fillna(
        #     value='Other_Null')

        # # 각 컬럼별로 Null값 건수 확인. 모두 0가 나와야 합니다.
        # print(df_train.isnull().sum())

        # df_test['brand_name'] = df_test['brand_name'].fillna(
        #     value='Other_Null')
        # df_test['category_name'] = df_test['category_name'].fillna(
        #     value='Other_Null')
        # df_test['item_description'] = df_test['item_description'].fillna(
        #     value='Other_Null')

        # # 각 컬럼별로 Null값 건수 확인. 모두 0가 나와야 합니다.
        # df_test.isnull().sum()

        # training_grouped = df_train.groupby(df_train.category_name)
        # testing_grouped = df_test.groupby(df_test.category_name)
        # print(training_grouped.dtypes)
        # print(testing_grouped.dtypes)

        # id_array = []
        # for t in training_grouped.category_name:
        #     id_array.append(t)

        # print(len(id_array))
        # print(id_array[0][0])
        # print(training_grouped.get_group(id_array[0][0]))

        # id_array_testing = []
        # for t in testing_grouped.category_name:
        #     id_array_testing.append(t)

        # print(len(id_array_testing))
        # print(id_array_testing[0][0])
        # print(testing_grouped.get_group(id_array_testing[0][0]))

        # df_array = {}
        # for t in id_array:
        #     df_array[t[0]] = (training_grouped.get_group(t[0]))

        # print(len(df_array))
        # print(df_array.keys())

        # df_array_testing = {}
        # for t in id_array_testing:
        #     df_array_testing[t[0]] = (testing_grouped.get_group(t[0]))

        # print(len(df_array_testing))
        # print(df_array_testing.keys())

        # targets = []
        # predictions = []
        # for category in df_array:

        #     train_cat = ""
        #     if category in df_array:
        #         train_cat = category
        #     else:
        #         split = category.split('/')
        #         train_cat = split[0] + '/' + split[1]

        #         for category in df_array:
        #             if category.find(train_cat) != -1:
        #                 train_cat = category
        #                 break

        #     print(train_cat)

        #     try:
        #         df = pd.DataFrame(df_array[train_cat])
        #         df_test = pd.DataFrame(df_array_testing[category])
        #     except:
        #         continue

        #     combo = pd.concat([df, df_test])

        #     # Convert "name" with feature vectorization
        #     cnt_vec = CountVectorizer(max_features=30000)
        #     combo_name = cnt_vec.fit(combo.name)
        #     X_name = cnt_vec.transform(df.name)
        #     Y_name = cnt_vec.transform(df_test.name)

        #     # Convert "item_description" with feature vectorization
        #     tfidf_descp = TfidfVectorizer(max_features=50000,
        #                                   ngram_range=(1, 3),
        #                                   stop_words='english')
        #     combo_desc = tfidf_descp.fit(combo['item_description'])
        #     X_descp = tfidf_descp.transform(df['item_description'])
        #     Y_descp = tfidf_descp.transform(df_test['item_description'])

        #     # Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
        #     lb_brand_name = LabelBinarizer(sparse_output=True)
        #     combo_brand = lb_brand_name.fit(combo['brand_name'])
        #     X_brand = lb_brand_name.transform(df['brand_name'])
        #     Y_brand = lb_brand_name.transform(df_test['brand_name'])

        #     lb_item_cond_id = LabelBinarizer(sparse_output=True)
        #     combo_item = lb_item_cond_id.fit(combo['item_condition_id'])
        #     X_item_cond_id = lb_item_cond_id.transform(df['item_condition_id'])
        #     Y_item_cond_id = lb_item_cond_id.transform(
        #         df_test['item_condition_id'])

        #     lb_shipping = LabelBinarizer(sparse_output=True)
        #     combo_ship = lb_shipping.fit(combo['shipping'])
        #     X_shipping = lb_shipping.transform(df['shipping'])
        #     Y_shipping = lb_shipping.transform(df_test['shipping'])

        #     # # RMSLE with Item Description
        #     # linear_model = Ridge(solver = "lsqr", fit_intercept=False)
        #     sparse_matrix_list_X = [
        #         X_descp, X_name, X_brand, X_item_cond_id, X_shipping
        #     ]
        #     sparse_matrix_list_Y = [
        #         Y_descp, Y_name, Y_brand, Y_item_cond_id, Y_shipping
        #     ]
        #     # linear_preds , y_test = model_train_predict(model=linear_model , matrix_list_X=sparse_matrix_list_X, matrix_list_Y=sparse_matrix_list_Y, df=df, df_test=df_test)

        #     lgbm_model = LGBMRegressor(n_estimators=50,
        #                                learning_rate=0.5,
        #                                num_leaves=125,
        #                                random_state=156)
        #     lgbm_preds, y_test = model_train_predict(
        #         model=lgbm_model,
        #         matrix_list_X=sparse_matrix_list_X,
        #         matrix_list_Y=sparse_matrix_list_Y,
        #         df=df,
        #         df_test=df_test)

        #     try:
        #         lgbm_preds = abs(lgbm_preds)

        #         # Compute median for this group and see if that produces a better error. If so use that.
        #         # median = math.floor(df['price'].median())
        #         # medians = []
        #         # for i in y_test:
        #         #     medians.append(median)
        #         # medians = np.array(medians)

        #         model_error = np.sqrt(
        #             np.mean(
        #                 np.power(np.log1p(lgbm_preds) - np.log1p(y_test), 2)))
        #         #median_error = np.sqrt(np.mean(np.power(np.log1p(medians) - np.log1p(y_test), 2)))
        #         print("Model ", model_error)
        #         #print("Median ", median_error)

        #         #if model_error < median_error:
        #         for pred in lgbm_preds:
        #             predictions.append(pred)
        #         # else:
        #         #     for median in medians:
        #         #         predictions.append(median)

        #         for target in y_test:
        #             targets.append(target)
        #     except:
        #         print("ERROR")

        # targets = np.array(targets)
        # predictions = np.array(predictions)
        # predictions = np.abs(predictions)
        # print(
        #     np.sqrt(
        #         np.mean(np.power(np.log1p(predictions) - np.log1p(targets),
        #                          2))))

    # model.preprocess()

    # sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping,
    #                       X_cat_dae, X_cat_jung, X_cat_so)
    # # X_features_sparse = hstack(sparse_matrix_list).tocsr()

    # # del X_features_sparse
    # gc.collect()

    # linear_model = Ridge(solver="lsqr", fit_intercept=False)

    # # RMSLE without Item Description
    # sparse_matrix_list = (X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae,
    #                       X_cat_jung, X_cat_so)
    # linear_preds, y_test = model_train_predict(model=linear_model,
    #                                            matrix_list=sparse_matrix_list)
    # print('Item Description을 제외했을 때 rmsle 값:',
    #       evaluate_org_price(y_test, linear_preds))

    # # RMSLE with Item Description
    # sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping,
    #                       X_cat_dae, X_cat_jung, X_cat_so)
    # linear_preds, y_test = model_train_predict(model=linear_model,
    #                                            matrix_list=sparse_matrix_list)
    # print('Item Description을 포함한 rmsle 값:',
    #       evaluate_org_price(y_test, linear_preds))

    # gc.collect()

    # sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping,
    #                       X_cat_dae, X_cat_jung, X_cat_so)

    # # RMSLE of LGBM
    # lgbm_model = LGBMRegressor(n_estimators=200,
    #                            learning_rate=0.5,
    #                            num_leaves=125,
    #                            random_state=156)
    # lgbm_preds, y_test = model_train_predict(model=lgbm_model,
    #                                          matrix_list=sparse_matrix_list)
    # print('LightGBM rmsle 값:', evaluate_org_price(y_test, lgbm_preds))

    # preds = lgbm_preds * 0.45 + linear_preds * 0.55
    # print('LightGBM과 Ridge를 ensemble한 최종 rmsle 값:',
    #       evaluate_org_price(y_test, preds))
