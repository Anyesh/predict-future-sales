import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .config.model_params import core_parameters
from .dispatcher import MODELS
from .feature_generator import FeatureGenerator
from .settings import CHK_PATH, OUTPUT_PATH, logger
from .utils import no_files, uniquify

warnings.filterwarnings("ignore")


def train_model(X_train, Y_train, X_validation, Y_validation, X_test, model_name):

    model = MODELS[model_name](
        **core_parameters[model_name],
    )

    mlflow.log_params(core_parameters[model_name])

    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_validation, Y_validation)],
        verbose=0,
        early_stopping_rounds=2,
        eval_metric="rmse",
    )

    return model


def test_model(X_train, Y_train, X_validation, Y_validation, X_test, model):
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    logger.info(
        f"Train rmse: {np.sqrt(mean_squared_error(Y_train, train_predictions))}"
    )
    mlflow.log_metric(
        "train_rmse", np.sqrt(mean_squared_error(Y_train, train_predictions))
    )
    logger.info(
        f"Validation rmse: {np.sqrt(mean_squared_error(Y_validation, validation_predictions))}"
    )
    mlflow.log_metric(
        "validation_rmse",
        np.sqrt(mean_squared_error(Y_validation, validation_predictions)),
    )

    return test_predictions


if __name__ == "__main__":
    fg = FeatureGenerator()
    model_name = "xgboost"

    if no_files(CHK_PATH, ["frozen_data.npz"]):
        X_train, Y_train, X_validation, Y_validation, X_test = fg.generate()
        X_train, Y_train, X_validation, Y_validation, X_test = (
            X_train.values,
            Y_train.values,
            X_validation.values,
            Y_validation.values,
            X_test.values,
        )
        np.savez(
            os.path.join(CHK_PATH, "frozen_data.npz"),
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_test=X_test,
        )
    else:
        logger.info("Loading data from checkpoint")
        _data = np.load(os.path.join(CHK_PATH, "frozen_data.npz"))
        X_train, Y_train, X_validation, Y_validation, X_test = (
            _data["X_train"],
            _data["Y_train"],
            _data["X_validation"],
            _data["Y_validation"],
            _data["X_test"],
        )

    if no_files(os.path.join(CHK_PATH, model_name, "model.pkl")):
        Path(os.path.join(CHK_PATH, model_name)).mkdir(parents=True, exist_ok=True)

        model = train_model(
            X_train, Y_train, X_validation, Y_validation, X_test, model_name=model_name
        )
        joblib.dump(model, uniquify(os.path.join(CHK_PATH, model_name, "model.pkl")))
    else:
        model = joblib.load(os.path.join(CHK_PATH, model_name, "model.pkl"))

    test_predictions = test_model(
        X_train, Y_train, X_validation, Y_validation, X_test, model
    )
    _, test_data = fg.get_datasets()
    prediction_df = pd.DataFrame(test_data["ID"], columns=["ID"])
    prediction_df["item_cnt_month"] = test_predictions.clip(0.0, 20.0)
    prediction_df.to_csv(
        uniquify(os.path.join(OUTPUT_PATH, f"{model_name}_submission.csv")), index=False
    )
