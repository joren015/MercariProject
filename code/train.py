import numpy as np
import pandas as pd

from model_logger import run

pd.set_option('max_colwidth', 200)
model_type = "NN"


def rmsle(y, y_pred):
    # underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle 계산
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))


if __name__ == "__main__":
    mercari_df = pd.read_csv('data/train.tsv', sep='\t', nrows=100)
    X = mercari_df[[x for x in mercari_df.columns if x != "price"]]
    y = mercari_df["price"]

    print("Preprocessing dataset")
    if model_type == "light_gbm":
        from light_gbm_model import LightGBMModel
        model = LightGBMModel()
        run(model, X, y, model.metrics, experiment=model.experiment, n_jobs=1)
    else:
        from nn_model import NNModel
        model = NNModel()
        model.my_evaluate(X, y)
        # model.preprocess()
        # X, y = model.apply_preprocessing(train)
        # run(model, X, y, model.metrics, experiment=model.experiment, n_jobs=1)

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
