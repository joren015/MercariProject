import math
import pickle
from os.path import exists

import mlflow
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (GRU, Dense, Dropout, Embedding, Flatten, Input,
                          concatenate)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from keras.metrics import LogCoshError, CosineSimilarity, MeanSquaredLogarithmicError, MeanAbsolutePercentageError, MeanAbsoluteError, RootMeanSquaredError, MeanSquaredError

from preprocessing import fit_and_save_vectorizer
from train import rmsle

pd.set_option('max_colwidth', 200)

# def rmsle(y, y_pred):
#     assert len(y) == len(y_pred)
#     to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))**2.0
#               for i, pred in enumerate(y_pred)]
#     return (sum(to_sum) * (1.0 / len(y)))**0.5


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


class NNModel(BaseEstimator, RegressorMixin):
    def __init__(self,
                 MAX_TEXT=259088,
                 MAX_CATEGORY=1311,
                 MAX_BRAND=5290,
                 MAX_CONDITION=6,
                 experiment="NN",
                 **kwarg):
        self.experiment = experiment
        self.model = self.get_model(MAX_TEXT, MAX_CATEGORY, MAX_BRAND,
                                    MAX_CONDITION)
        self.vectorizers = {}
        for col in [
                "category_name", "brand_name", "item_description_and_name"
        ]:
            if not exists("data/nn/{}_vectorizer/vectorizer.pkl".format(col)):
                self.preprocess()

            with open("data/nn/{}_vectorizer/vectorizer.pkl".format(col),
                      'rb') as f:
                self.vectorizers[col] = pickle.load(f)

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

    #KERAS MODEL DEFINITION
    def get_model(self, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION):
        #params
        dr_r = 0.1

        #Inputs
        name = Input(shape=[10], name="name")
        # name = Input(shape=[X_train["name"].shape[1]], name="name")
        item_desc = Input(shape=[75], name="item_desc")
        # item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
        brand_name = Input(shape=[1], name="brand_name")
        category_name = Input(shape=[1], name="category_name")
        item_condition = Input(shape=[1], name="item_condition")
        num_vars = Input(shape=[1], name="num_vars")
        # num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

        #Embeddings layers
        emb_name = Embedding(MAX_TEXT, 50)(name)
        emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
        emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
        emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
        emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

        #rnn layer
        rnn_layer1 = GRU(16)(emb_item_desc)
        rnn_layer2 = GRU(8)(emb_name)

        #main layer
        main_l = concatenate([
            Flatten()(emb_brand_name),
            Flatten()(emb_category_name),
            Flatten()(emb_item_condition), rnn_layer1, rnn_layer2, num_vars
        ])
        main_l = Dropout(dr_r)(Dense(128)(main_l))
        main_l = Dropout(dr_r)(Dense(64)(main_l))

        #output
        output = Dense(1, activation="linear")(main_l)

        #model
        model = Model([
            name, item_desc, brand_name, category_name, item_condition,
            num_vars
        ], output)
        model.compile(loss="mse",
                      optimizer="adam",
                      metrics=[
                          "mae", self.rmsle_cust,
                          LogCoshError(),
                          CosineSimilarity(),
                          MeanSquaredLogarithmicError(),
                          MeanAbsolutePercentageError(),
                          MeanAbsoluteError(),
                          RootMeanSquaredError(),
                          MeanSquaredError()
                      ])

        return model

    def fit(self, X, y):
        BATCH_SIZE = 20000
        epochs = 5
        X = self.apply_preprocessing(X)
        y = np.log1p(y)
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            #    validation_data=(X_valid, y_valid),
            verbose=1)

        # #EVLUEATE THE MODEL ON DEV TEST: What is it doing?
        # val_preds = model.predict(X_valid)
        # #val_preds = target_scaler.inverse_transform(val_preds)
        # val_preds = np.exp(val_preds) + 1

        # #mean_absolute_error, mean_squared_log_error
        # y_true = np.array(dvalid.price.values)
        # y_pred = val_preds[:, 0]
        # v_rmsle = rmsle(y_true, y_pred)
        # print(" RMSLE error on dev test: " + str(v_rmsle))

    def predict(self, X):
        X = self.apply_preprocessing(X)
        y_preds = self.model.predict(X)
        return np.expm1(y_preds)

    def rmsle_cust(self, y_true, y_pred):
        y_true = K.exp(y_true)
        y_pred = K.exp(y_pred)
        first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
        return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

    def score(self, y, y_pred):
        return -1 * self.rmsle_cust(y, y_pred)

    def get_params(self, deep=True):
        return self.model.get_config()

    def common_preprocessing(self, dataset):
        dataset.category_name.fillna(value="missing", inplace=True)
        dataset.brand_name.fillna(value="missing", inplace=True)
        dataset.item_description.fillna(value="missing", inplace=True)
        return (dataset)

    def preprocess(self, include_analysis=True):
        print("Loading data...")
        train = pd.read_table("data/train.tsv")
        test = pd.read_table("data/test.tsv")
        if include_analysis:
            print(train.shape)
            print(test.shape)

        print("Handling missing values...")

        train = self.common_preprocessing(train)
        test = self.common_preprocessing(test)
        if include_analysis:
            print(train.shape)
            print(test.shape)

        #PROCESS CATEGORICAL DATA
        print("Handling categorical variables...")

        le = fit_and_save_vectorizer(
            np.hstack([train.category_name, test.category_name]),
            LabelEncoder(), "data/nn/category_name_vectorizer")
        train.category_name = le.transform(train.category_name)
        test.category_name = le.transform(test.category_name)

        le = fit_and_save_vectorizer(
            np.hstack([train.brand_name, test.brand_name]), LabelEncoder(),
            "data/nn/brand_name_vectorizer")

        train.brand_name = le.transform(train.brand_name)
        test.brand_name = le.transform(test.brand_name)
        del le

        # train.head(3)

        #PROCESS TEXT: RAW
        print("Text to seq process...")

        # raw_text = np.hstack(
        #     [train.item_description.str.lower(),
        #      train.name.str.lower()])

        print("   Fitting tokenizer...")
        # tok_raw = Tokenizer()
        # tok_raw.fit_on_texts(raw_text)
        tok_raw = fit_and_save_vectorizer(
            np.hstack(
                [train.item_description.str.lower(),
                 train.name.str.lower()]),
            Tokenizer(),
            "data/nn/item_description_and_name_vectorizer",
            non_sklearn=True)
        print("   Transforming text to seq...")

        train["seq_item_description"] = tok_raw.texts_to_sequences(
            train.item_description.str.lower())
        test["seq_item_description"] = tok_raw.texts_to_sequences(
            test.item_description.str.lower())
        train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
        test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
        # train.head(3)

        #SEQUENCES VARIABLES ANALYSIS
        if include_analysis:
            max_name_seq = np.max([
                np.max(train.seq_name.apply(lambda x: len(x))),
                np.max(test.seq_name.apply(lambda x: len(x)))
            ])
            max_seq_item_description = np.max([
                np.max(train.seq_item_description.apply(lambda x: len(x))),
                np.max(test.seq_item_description.apply(lambda x: len(x)))
            ])
            print("max name seq " + str(max_name_seq))
            print("max item desc seq " + str(max_seq_item_description))

            train.seq_name.apply(lambda x: len(x)).hist()
            train.seq_item_description.apply(lambda x: len(x)).hist()

        #EMBEDDINGS MAX VALUE
        #Base on the histograms, we select the next lengths
        MAX_NAME_SEQ = 10
        MAX_ITEM_DESC_SEQ = 75
        MAX_TEXT = np.max([
            np.max(train.seq_name.max()),
            np.max(test.seq_name.max()),
            np.max(train.seq_item_description.max()),
            np.max(test.seq_item_description.max())
        ]) + 2
        MAX_CATEGORY = np.max(
            [train.category_name.max(),
             test.category_name.max()]) + 1
        MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()]) + 1
        MAX_CONDITION = np.max(
            [train.item_condition_id.max(),
             test.item_condition_id.max()]) + 1

        if include_analysis:
            print("MAX_TEXT: {}".format(MAX_TEXT))
            print("MAX_CATEGORY: {}".format(MAX_CATEGORY))
            print("MAX_BRAND: {}".format(MAX_BRAND))
            print("MAX_CONDITION: {}".format(MAX_CONDITION))

        #SCALE target variable
        # train["target"] = np.log(train.price + 1)
        # target_scaler = MinMaxScaler(feature_range=(-1, 1))
        #train["target"] = target_scaler.fit_transform(train.target.reshape(-1,1))

        train = {
            'name':
            pad_sequences(train.seq_name, maxlen=MAX_NAME_SEQ),
            'item_desc':
            pad_sequences(train.seq_item_description,
                          maxlen=MAX_ITEM_DESC_SEQ),
            'brand_name':
            np.array(train.brand_name),
            'category_name':
            np.array(train.category_name),
            'item_condition':
            np.array(train.item_condition_id),
            'num_vars':
            np.array(train[["shipping"]])
        }

        if include_analysis:
            print("MAX_TEXT: {}".format(MAX_TEXT))
            print("MAX_CATEGORY: {}".format(MAX_CATEGORY))
            print("MAX_BRAND: {}".format(MAX_BRAND))
            print("MAX_CONDITION: {}".format(MAX_CONDITION))
            print(train["name"].shape[1])
            print(train["item_desc"].shape[1])
            print(train["num_vars"].shape[1])

        if include_analysis:
            print(train)
            # pd.DataFrame(train["price"]).hist()

        # train.to_csv("data/nn/nn_dataset_train.csv")
        return MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION

        #EXTRACT DEVELOPTMENT TEST
        # dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
        # print(dtrain.shape)
        # print(dvalid.shape)

        #KERAS DATA DEFINITION

        # X_train = get_keras_data(dtrain)
        # X_valid = get_keras_data(dvalid)
        # X_test = get_keras_data(test)

    def apply_preprocessing(self, X, MAX_NAME_SEQ=10, MAX_ITEM_DESC_SEQ=75):
        X = self.common_preprocessing(X)

        X.category_name = self.vectorizers["category_name"].transform(
            X.category_name)
        X.brand_name = self.vectorizers["brand_name"].transform(X.brand_name)

        # X.category_name = load_and_apply_vectorizer(
        #     "data/nn/category_name_label_encoder/vectorizer.pkl",
        #     X.category_name)
        # X.brand_name = load_and_apply_vectorizer(
        #     "data/nn/brand_name_label_encoder/vectorizer.pkl", X.brand_name)
        # train.brand_name = le.transform(train.category_name)
        # train.brand_name = le.transform(train.brand_name)

        self.vectorizers["item_description_and_name"].texts_to_sequences(
            X.item_description.str.lower())

        self.vectorizers["item_description_and_name"].texts_to_sequences(
            X.name.str.lower())

        with open(
                "data/nn/item_description_and_name_vectorizer/vectorizer.pkl",
                'rb') as f:
            vectorizer = pickle.load(f)

        X["seq_item_description"] = vectorizer.texts_to_sequences(
            X.item_description.str.lower())
        X["seq_name"] = vectorizer.texts_to_sequences(X.name.str.lower())
        # train["seq_item_description"] = tok_raw.texts_to_sequences(
        #     train.item_description.str.lower())
        # test["seq_item_description"] = tok_raw.texts_to_sequences(
        #     test.item_description.str.lower())
        # train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
        # test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())

        # y = np.log(X.price + 1)
        # train["target"] = np.log(train.price + 1)
        X = {
            'name':
            pad_sequences(X.seq_name, maxlen=MAX_NAME_SEQ),
            'item_desc':
            pad_sequences(X.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
            'brand_name':
            np.array(X.brand_name),
            'category_name':
            np.array(X.category_name),
            'item_condition':
            np.array(X.item_condition_id),
            'num_vars':
            np.array(X[["shipping"]])
        }

        return X

    def my_evaluate(self, X, y, n_splits=10, n_repeats=5, n_jobs=1):
        n_jobs = 1
        experiment_names = [x.name for x in mlflow.list_experiments()]
        if self.experiment not in experiment_names:
            mlflow.create_experiment(self.experiment)

        experiment_id = mlflow.get_experiment_by_name(
            self.experiment).experiment_id
        mlflow.keras.autolog()
        with mlflow.start_run(experiment_id=experiment_id) as run:
            cv = RepeatedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=42)
            for train_index, test_index in cv.split(X):
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]
                # X_train, X_test = {k: v[train_index]
                #                    for k, v in X.items()
                #                    }, {k: v[test_index]
                #                        for k, v in X.items()}
                # y_train, y_test = y[train_index], y[test_index]
                print(len(X_train))
                print(len(y_train))
                print(len(X_test))
                print(len(y_test))
                mlflow.log_params({
                    "train_size": len(y_train),
                    "test_size": len(y_test)
                })

                self.fit(X_train, y_train)
                X_test = self.apply_preprocessing(X_test)
                y_test = np.log1p(y_test)
                results = self.model.evaluate(X_test,
                                              y_test,
                                              batch_size=1000,
                                              return_dict=True)
                results = {"test_{}".format(k): v for k, v in results.items()}
                mlflow.log_metrics(results)

            # results = cross_validate(self.model,
            #                          X,
            #                          y,
            #                          scoring=self.metrics,
            #                          cv=cv,
            #                          n_jobs=n_jobs,
            #                          return_estimator=True,
            #                          verbose=1)

            # results["fit_time"] = [np.double(x) for x in results["fit_time"]]
            # for k, v in results.items():
            #     for i in range(len(v)):
            #         if k == "estimator":
            #             mlflow.log_params(v[i].get_params(deep=True))
            #         else:
            #             mlflow.log_metric(k, v[i])
