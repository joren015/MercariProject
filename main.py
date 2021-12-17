import argparse

import pandas as pd

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
    else:
        model.my_evaluate(X, y)