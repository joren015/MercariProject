import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_type",
    help=
    "Which model to use for the given task. Options: light_gbm, neural_network, category_model, ridge_regression.",
    type=str)
parser.add_argument(
    "task",
    help="Task to perform with chosen model. Options: evaluate, predict.",
    type=str)

args = parser.parse_args()

if __name__ == "__main__":
    df_train = pd.read_csv('data/train.tsv', sep='\t')

    X = df_train
    y = df_train["price"]

    if args.model_type == "light_gbm":
        from app.light_gbm_model import LightGBMModel
        model = LightGBMModel()
    elif args.model_type == "neural_network":
        from app.nn_model import NNModel
        model = NNModel()
    elif args.model_type == "category_model":
        from app.category_model import CategoryModel
        model = CategoryModel()
    else:
        from app.ridge_regression_model import RidgeRegressionModel
        model = RidgeRegressionModel()

    if args.task == "predict":
        model.fit(X, y)
        df_eval = pd.read_csv('data/test_stg2.tsv', sep='\t')
        y_hat = model.predict(df_eval)
        df_submit = df_eval[["test_id"]].copy(deep=True)
        df_submit["price"] = y_hat.round(3)
        df_submit.to_csv("./submission.csv", index=False)
    elif args.task == "evaluate":
        model.my_evaluate(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.1)
        model.fit(X_train, y_train)
        df_eval = X_test
        y_hat = model.predict(df_eval)
        df_eval["y_hat"] = y_hat
        err = np.sqrt(
            np.mean(np.power(np.log1p(y_test) - np.log1p(y_hat.round(3)), 2)))
        print(err)
        df_eval["err"] = np.power(
            np.log1p(y_test) - np.log1p(y_hat.round(3)), 2)
        print(np.sqrt(np.mean(df_eval["err"])))
        X_train.to_csv("./train.csv")
        df_eval.to_csv("./eval.csv")
