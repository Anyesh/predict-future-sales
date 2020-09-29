import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .config.model_params import core_parameters, model_params
from .dispatcher import MODELS
from .feature_generator import FeatureGenerator
from .settings import CHK_PATH, OUTPUT_PATH, logger
from .utils import get_latest_file, no_files, uniquify

warnings.filterwarnings("ignore")


def train_model(X_train, Y_train, X_validation, Y_validation, X_test, model_name):

    model = MODELS[model_name](
        **core_parameters[model_name],
    )

    mlflow.log_params(core_parameters[model_name])
    mlflow.log_param("model_name", model_name)

    model.fit(
        X_train,
        Y_train,
        # eval_set=[(X_validation, Y_validation)],
        **model_params[model_name],
    )

    return model


def test_model(X_train, Y_train, X_validation, Y_validation, X_test, model):
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(Y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(Y_validation, validation_predictions))

    logger.info(f"Train rmse: {train_rmse}")
    mlflow.log_metric("train_rmse", train_rmse)
    logger.info(f"Validation rmse: {val_rmse}")
    mlflow.log_metric("validation_rmse", val_rmse)

    return test_predictions


if __name__ == "__main__":
    MODEL_NAME = "randomforest"
    TRAIN_MODEL = True
    IGNORE_CHECKPOINT = False

    fg = FeatureGenerator()

    if no_files(CHK_PATH, ["frozen_data.npz"]) or IGNORE_CHECKPOINT:
        X_train, Y_train, X_validation, Y_validation, test_data, cols = fg.generate()
        logger.info(f"Final data columns {cols}")

        X_train, Y_train, X_validation, Y_validation, X_test = (
            X_train.values,
            Y_train.values,
            X_validation.values,
            Y_validation.values,
            test_data[cols].values,
        )
        index = test_data.index
        np.savez(
            os.path.join(CHK_PATH, "frozen_data.npz"),
            X_train=X_train,
            Y_train=Y_train,
            X_validation=X_validation,
            Y_validation=Y_validation,
            X_test=X_test,
            index=index,
        )
    else:
        logger.info("Loading data from checkpoint")
        _ = np.load(os.path.join(CHK_PATH, "frozen_data.npz"))
        X_train, Y_train, X_validation, Y_validation, X_test, index = (
            _["X_train"],
            _["Y_train"],
            _["X_validation"],
            _["Y_validation"],
            _["X_test"],
            _["index"],
        )

    Path(os.path.join(CHK_PATH, MODEL_NAME)).mkdir(parents=True, exist_ok=True)
    if no_files(os.path.join(CHK_PATH, MODEL_NAME), ["model.pkl"]) or TRAIN_MODEL:
        logger.info(f"{MODEL_NAME} model training started")

        model = train_model(
            X_train, Y_train, X_validation, Y_validation, X_test, model_name=MODEL_NAME
        )

        joblib.dump(model, uniquify(os.path.join(CHK_PATH, MODEL_NAME, "model.pkl")))
    else:
        latest_model = get_latest_file(os.path.join(CHK_PATH, MODEL_NAME), "*.pkl")
        logger.info(f"Using latest {MODEL_NAME} model: {str(latest_model)}")

        model = joblib.load(latest_model)

    logger.info(str(model))

    test_predictions = test_model(
        X_train, Y_train, X_validation, Y_validation, X_test, model
    )

    prediction_df = pd.DataFrame(index, columns=["ID"])
    prediction_df["item_cnt_month"] = test_predictions.clip(0.0, 20.0)

    pred_file_name = uniquify(os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_submission.csv"))
    mlflow.log_param("file_name", pred_file_name)
    prediction_df.to_csv(pred_file_name, index=False)
