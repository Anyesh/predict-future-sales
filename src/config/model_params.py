core_parameters = {
    "xgboost": {
        "max_depth": 8,
        "n_estimators": 500,
        "min_child_weight": 1000,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "eta": 0.3,
        "seed": 0,
        "gpu_id": 0,
    },
    "linear_regression": {},
    "extratrees": {
        "n_estimators": 25,
        "n_jobs": -1,
        "max_depth": 20,
        "random_state": 0,
    },
    "randomforest": {
        "n_estimators": 50,
        "max_depth": 7,
        "random_state": 0,
        "n_jobs": -1,
    },
}

model_params = {
    "xgboost": {
        "verbose": 0,
        # "early_stopping_rounds": 20,
        # "eval_metric": "rmse",
    },
    "randomforest": {},
    "extratrees": {},
    "linear_regression": {},
}
