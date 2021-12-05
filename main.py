import argparse

import pandas as pd

from app.train import main

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_type",
    help=
    "Which model to use for the given task. Options: light_gbm, neural_network, category_model.",
    type=str)
parser.add_argument(
    "task",
    help="Task to perform with chosen model. Options: evaluate, predict.",
    type=str)

args = parser.parse_args()

if __name__ == "__main__":
    df_train = pd.read_csv('data/train.tsv', sep='\t', nrows=10000)

    X = df_train
    y = df_train["price"]

    if args.model_type == "light_gbm":
        from app.light_gbm_model import LightGBMModel
        model = LightGBMModel()
    elif args.model_type == "neural_network":
        from app.nn_model import NNModel
        model = NNModel()
    else:
        from app.category_model import CategoryModel
        model = CategoryModel()

    if args.task == "predict":
        model.fit(X, y)
        df_eval = pd.read_csv('data/test.tsv', sep='\t', nrows=1000)
        y_hat = model.predict(df_eval)
        df_submit = df_eval[["test_id"]].copy(deep=True)
        df_submit["price"] = y_hat.round(3)
        df_submit.to_csv("./submit.csv", index=False)
    else:
        main(args.model_type)
