import mlflow
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate


def run(model, X, y, metrics, experiment, n_splits=2, n_repeats=2, n_jobs=1):
    n_jobs = 1
    experiment_names = [x.name for x in mlflow.list_experiments()]
    if experiment not in experiment_names:
        mlflow.create_experiment(experiment)

    experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        cv = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats,
                           random_state=42)
        results = cross_validate(model,
                                 X,
                                 y,
                                 scoring=metrics,
                                 cv=cv,
                                 n_jobs=n_jobs,
                                 return_estimator=True,
                                 verbose=1)

        results["fit_time"] = [np.double(x) for x in results["fit_time"]]
        for k, v in results.items():
            for i in range(len(v)):
                if k == "estimator":
                    mlflow.log_params(v[i].get_params(deep=True))
                else:
                    mlflow.log_metric(k, v[i])
