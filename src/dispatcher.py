from sklearn import ensemble
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

MODELS = {
    "randomforest": ensemble.RandomForestRegressor,
    "extratrees": ensemble.ExtraTreesRegressor,
    "xgboost": XGBRegressor,
    "linear_regression": LinearRegression,
}
