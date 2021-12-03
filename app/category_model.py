import pickle
from uuid import uuid4

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import make_scorer
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
        # self.vectorizers = {}
        # for col in [
        #         "name", "item_description", "brand_name", "item_condition_id",
        #         "shipping", "cat_dae", "cat_jung", "cat_so"
        # ]:
        #     if not exists(
        #             "data/light_gbm/{}_vectorizer/vectorizer.pkl".format(col)):
        #         if quick_preprocess:
        #             self.preprocess(10000)
        #         else:
        #             self.preprocess()

        #     self.vectorizers[
        #         col] = "data/light_gbm/{}_vectorizer/vectorizer.pkl".format(
        #             col)

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

    def fit(self, X):
        self.models = {}
        self.vectorizer_guid = str(uuid4()).replace('-', '_')
        Xp = X.copy(deep=True)
        self.preprocess(Xp)
        del Xp
        X = self.apply_preprocessing(X)
        for category in self.trained_categories:
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
                # lgbm_preds, y_test, model = self.model_train_predict(
                #     model=lgbm_model,
                #     matrix_list_X=sparse_matrix_list_X,
                #     matrix_list_Y=sparse_matrix_list_Y,
                #     df=df,
                #     df_test=df_test)

            # try:
            #     lgbm_preds = abs(lgbm_preds)

            #     model_error = np.sqrt(
            #         np.mean(np.power(np.log1p(lgbm_preds) - np.log1p(y_test), 2)))
            #     #median_error = np.sqrt(np.mean(np.power(np.log1p(medians) - np.log1p(y_test), 2)))
            #     print("Model ", model_error)
            #     #print("Median ", median_error)

            #     #if model_error < median_error:
            #     for pred in lgbm_preds:
            #         predictions.append(pred)
            #     # else:
            #     #     for median in medians:
            #     #         predictions.append(median)

            #     for target in y_test:
            #         targets.append(target)
            # except:
            #     print("ERROR")

            # self.model.fit(X, y)

    def predict(self, X):
        X = self.apply_preprocessing(X)
        y_preds = self.model.predict(X)
        return np.expm1(y_preds)

    def score(self, y, y_pred):
        return -1 * rmsle(y, y_pred)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def model_train_predict(self, model, matrix_list_X, matrix_list_Y, df,
                            df_test):
        # scipy.sparse 모듈의 hstack 을 이용하여 sparse matrix 결합
        #
        X = hstack(matrix_list_X).tocsr()
        Y = hstack(matrix_list_Y).tocsr()
        # X = matrix_list_X
        # Y = matrix_list_Y

        # X_train, X_test, y_train, y_test=train_test_split(X, df['price'],
        #                                                   test_size=0.1)

        # Y_train, Y_test, y_price, y_useless=train_test_split(Y, df_test['price'],
        #                                                   test_size=0.1)

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
        df_array = self.common_preprocessing(X)

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
            # cnt_vec = CountVectorizer(max_features=30000)
            # combo_name = cnt_vec.fit(combo.name)
            # self.models[category]["vectorizers"]["name"] = combo_name

            fit_and_save_vectorizer(
                combo.name, CountVectorizer(max_features=30000),
                "data/category_model/{}/name_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "name"] = "data/category_model/{}/name_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            # X_name = cnt_vec.transform(df.name)
            # Y_name = cnt_vec.transform(df_test.name)

            # Convert "item_description" with feature vectorization
            # tfidf_descp = TfidfVectorizer(max_features=50000,
            #                               ngram_range=(1, 3),
            #                               stop_words='english')
            # combo_desc = tfidf_descp.fit(combo['item_description'])
            # self.models[category]["vectorizers"][
            #     "item_description"] = combo_desc

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

            # X_descp = tfidf_descp.transform(df['item_description'])
            # Y_descp = tfidf_descp.transform(df_test['item_description'])

            # Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
            # lb_brand_name = LabelBinarizer(sparse_output=True)
            # combo_brand = lb_brand_name.fit(combo['brand_name'])
            # self.models[category]["vectorizers"]["brand_name"] = combo_brand

            fit_and_save_vectorizer(
                combo["brand_name"], LabelBinarizer(sparse_output=True),
                "data/category_model/{}/brand_name_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "brand_name"] = "data/category_model/{}/brand_name_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            # X_brand = lb_brand_name.transform(df['brand_name'])
            # Y_brand = lb_brand_name.transform(df_test['brand_name'])

            # lb_item_cond_id = LabelBinarizer(sparse_output=True)
            # combo_item = lb_item_cond_id.fit(combo['item_condition_id'])
            # self.models[category]["vectorizers"][
            #     "item_condition_id"] = combo_item

            fit_and_save_vectorizer(
                combo["item_condition_id"], LabelBinarizer(sparse_output=True),
                "data/category_model/{}/item_condition_id_vectorizer".format(
                    self.vectorizer_guid))
            self.models[category]["vectorizers"][
                "item_condition_id"] = "data/category_model/{}/item_condition_id_vectorizer/vectorizer.pkl".format(
                    self.vectorizer_guid)

            # X_item_cond_id = lb_item_cond_id.transform(df['item_condition_id'])
            # Y_item_cond_id = lb_item_cond_id.transform(
            #     df_test['item_condition_id'])

            # lb_shipping = LabelBinarizer(sparse_output=True)
            # combo_ship = lb_shipping.fit(combo['shipping'])
            # self.models[category]["vectorizers"]["shipping"] = combo_ship

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
        training_grouped = X.groupby(X.category_name)
        print(training_grouped.dtypes)

        id_array = []
        for t in training_grouped.category_name:
            id_array.append(t)

        df_array = {}
        for t in id_array:
            df_array[t[0]] = (training_grouped.get_group(t[0]))

        return df_array

    def apply_preprocessing(self, X):
        df_array = self.common_preprocessing(X)

        sparse_matrix_dict = {}
        for category, _ in df_array.items():
            print("Preprocessing {}".format(category))
            df = pd.DataFrame(df_array[category])
            sparse_matrix_list = []
            for col in [
                    "name", "item_description", "brand_name",
                    "item_condition_id", "shipping"
            ]:
                v = self.models[category]["vectorizers"][col]
                with open(v, 'rb') as f:
                    vectorizer = pickle.load(f)
                    if col != "item_description":
                        X.loc[~X[col].isin(vectorizer.classes_),
                              col] = "Other_Null"

                    sparse_matrix_list.append(vectorizer.transform(df[col]))

            sparse_matrix_dict[category] = {
                "df": df,
                "x": hstack(sparse_matrix_list).tocsr()
            }

        return sparse_matrix_dict
