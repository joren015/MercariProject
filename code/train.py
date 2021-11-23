import math

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (GRU, Dense, Dropout, Embedding, Flatten, Input,
                          concatenate)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from light_gbm_model import LightGBMModel
from model_logger import run
from preprocessing import preprocess_light_gbm

pd.set_option('max_colwidth', 200)


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))**2.0
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y)))**0.5


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_keras_data(dataset):
    X = {
        'name':
        pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc':
        pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name':
        np.array(dataset.brand_name),
        'category_name':
        np.array(dataset.category_name),
        'item_condition':
        np.array(dataset.item_condition_id),
        'num_vars':
        np.array(dataset[["shipping"]])
    }
    return X


#KERAS MODEL DEFINITION
def get_model():
    #params
    dr_r = 0.1

    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

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
    model = Model(
        [name, item_desc, brand_name, category_name, item_condition, num_vars],
        output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model


if __name__ == "__main__":
    print("Loading data...")
    train = pd.read_table("data/train.tsv")
    test = pd.read_table("data/test.tsv")
    print(train.shape)
    print(test.shape)

    print("Handling missing values...")

    train = handle_missing(train)
    test = handle_missing(test)
    print(train.shape)
    print(test.shape)

    #PROCESS CATEGORICAL DATA
    print("Handling categorical variables...")
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    del le

    train.head(3)

    #PROCESS TEXT: RAW
    print("Text to seq process...")

    raw_text = np.hstack(
        [train.item_description.str.lower(),
         train.name.str.lower()])

    print("   Fitting tokenizer...")
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    print("   Transforming text to seq...")

    train["seq_item_description"] = tok_raw.texts_to_sequences(
        train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(
        test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
    train.head(3)

    #SEQUENCES VARIABLES ANALYSIS
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

    #SCALE target variable
    train["target"] = np.log(train.price + 1)
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    #train["target"] = target_scaler.fit_transform(train.target.reshape(-1,1))
    pd.DataFrame(train.target).hist()

    #EXTRACT DEVELOPTMENT TEST
    dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
    print(dtrain.shape)
    print(dvalid.shape)

    #KERAS DATA DEFINITION

    X_train = get_keras_data(dtrain)
    X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test)

    model = get_model()
    model.summary()

    #FITTING THE MODEL
    BATCH_SIZE = 20000
    epochs = 5

    model = get_model()
    model.fit(X_train,
              dtrain.target,
              epochs=epochs,
              batch_size=BATCH_SIZE,
              validation_data=(X_valid, dvalid.target),
              verbose=1)

    #EVLUEATE THE MODEL ON DEV TEST: What is it doing?
    val_preds = model.predict(X_valid)
    #val_preds = target_scaler.inverse_transform(val_preds)
    val_preds = np.exp(val_preds) + 1

    #mean_absolute_error, mean_squared_log_error
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: " + str(v_rmsle))

    print(1 / 0)
    print("Preprocessing dataset")
    preprocess_light_gbm(nrows=1000)
    model = LightGBMModel()
    X, y = model.get_dataset(nrows=1000)

    run(model, X, y, model.metrics, experiment=model.experiment, n_jobs=1)

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
