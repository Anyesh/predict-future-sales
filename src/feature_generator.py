import os

import numpy as np
import pandas as pd

from . import settings
from .settings import logger
from .utils import download_data, no_files


class FeatureGenerator:
    def __init__(self):
        self.download_data()

    def download_data(self):
        if no_files(settings.DATA_DIR, settings.DATA_NAMES):
            for name in settings.DATA_NAMES:
                _ = download_data(
                    str(settings.DATA_URL) + "/" + str(name), settings.DATA_DIR
                )
        logger.info(f"Using {' '.join(settings.DATA_NAMES)} data files")

    def __call__(self):
        pass

    def get_datasets(self):
        test_data = pd.read_csv(
            os.path.join(settings.DATA_DIR, "test.csv"),
            dtype={"ID": "int32", "shop_id": "int32", "item_id": "int32"},
        )
        item_categories = pd.read_csv(
            os.path.join(settings.DATA_DIR, "item_categories.csv"),
            dtype={"item_category_name": "str", "item_category_id": "int32"},
        )
        items = pd.read_csv(
            os.path.join(settings.DATA_DIR, "items.csv"),
            dtype={"item_name": "str", "item_id": "int32", "item_category_id": "int32"},
        )
        shops = pd.read_csv(
            os.path.join(settings.DATA_DIR, "shops.csv"),
            dtype={"shop_name": "str", "shop_id": "int32"},
        )
        sales = pd.read_csv(
            os.path.join(settings.DATA_DIR, "sales_train.csv"),
            parse_dates=["date"],
            dtype={
                "date": "str",
                "date_block_num": "int32",
                "shop_id": "int32",
                "item_id": "int32",
                "item_price": "float32",
                "item_cnt_day": "int32",
            },
        )

        logger.info("Joining items, shops and item_categories to sales")

        train_data = (
            sales.join(items, on="item_id", rsuffix="_")
            .join(shops, on="shop_id", rsuffix="_")
            .join(item_categories, on="item_category_id", rsuffix="_")
            .drop(["item_id_", "shop_id_", "item_category_id_"], axis=1)
        )

        return train_data, test_data

    def clean_data(self, train_data, test_data):
        logger.info("Start data cleaning")

        train_data = train_data[
            (train_data["item_price"] > 0) & (train_data["item_cnt_day"] > 0)
        ]

        test_shop_ids = test_data["shop_id"].unique()
        test_item_ids = test_data["item_id"].unique()

        lk_train = train_data[train_data["shop_id"].isin(test_shop_ids)]

        lk_train = lk_train[lk_train["item_id"].isin(test_item_ids)]

        train_monthly = lk_train[
            [
                "date",
                "date_block_num",
                "shop_id",
                "item_category_id",
                "item_id",
                "item_price",
                "item_cnt_day",
            ]
        ]
        train_monthly = (
            train_monthly.sort_values("date")
            .groupby(
                ["date_block_num", "shop_id", "item_category_id", "item_id"],
                as_index=False,
            )
            .agg(
                {
                    "item_price": ["sum", "mean"],
                    "item_cnt_day": ["sum", "mean", "count"],
                }
            )
        )

        train_monthly.columns = [
            "date_block_num",
            "shop_id",
            "item_category_id",
            "item_id",
            "item_price",
            "mean_item_price",
            "item_cnt",
            "mean_item_cnt",
            "transactions",
        ]
        logger.info("End data cleaning")

        return train_monthly

    def fill_data(self, monthly_data):
        logger.info("Start data filling")

        shop_ids = monthly_data["shop_id"].unique()
        item_ids = monthly_data["item_id"].unique()

        empty_df = []
        for i in range(34):  # upto 33
            for shop in shop_ids:
                for item in item_ids:
                    empty_df.append([i, shop, item])

        empty_df = pd.DataFrame(
            empty_df, columns=["date_block_num", "shop_id", "item_id"]
        )
        monthly_data = pd.merge(
            empty_df,
            monthly_data,
            on=["date_block_num", "shop_id", "item_id"],
            how="left",
        )
        monthly_data.fillna(0, inplace=True)
        logger.info("End data filling")

        return monthly_data

    def group_data(self, monthly_data):
        ## item sale's mean and total
        gp_month_mean = (
            monthly_data[monthly_data["year"] == 2013]
            .groupby(["month"], as_index=False)["item_cnt"]
            .mean()
        )
        gp_month_sum = (
            monthly_data[monthly_data["year"] == 2013]
            .groupby(["month"], as_index=False)["item_cnt"]
            .sum()
        )

        ## item categories' mean and total
        gp_category_mean = monthly_data.groupby(["item_category_id"], as_index=False)[
            "item_cnt"
        ].mean()
        gp_category_sum = monthly_data.groupby(["item_category_id"], as_index=False)[
            "item_cnt"
        ].sum()

        ## item shop_id's mean and total
        gp_shop_mean = monthly_data.groupby(["shop_id"], as_index=False)[
            "item_cnt"
        ].mean()
        gp_shop_sum = monthly_data.groupby(["shop_id"], as_index=False)[
            "item_cnt"
        ].sum()

        return (
            gp_month_mean,
            gp_month_sum,
            gp_category_mean,
            gp_category_sum,
            gp_shop_mean,
            gp_shop_sum,
        )

    # def generate_rolling(self, train_monthly):
    #     logger.info("Start generating rolling data")

    #     # Min value
    #     f_min = lambda x: x.rolling(window=3, min_periods=1).min()
    #     # Max value
    #     f_max = lambda x: x.rolling(window=3, min_periods=1).max()
    #     # Mean value
    #     f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()
    #     # Standard deviation
    #     f_std = lambda x: x.rolling(window=3, min_periods=1).std()

    #     function_list = [f_min, f_max, f_mean, f_std]
    #     function_name = ["min", "max", "mean", "std"]

    #     for i in range(len(function_list)):
    #         train_monthly[("item_cnt_%s" % function_name[i])] = (
    #             train_monthly.sort_values("date_block_num")
    #             .groupby(["shop_id", "item_category_id", "item_id"])["item_cnt"]
    #             .apply(function_list[i])
    #         )

    #     # Fill the empty std features with 0
    #     train_monthly["item_cnt_std"].fillna(0, inplace=True)

    #     lag_list = [1, 2, 3]
    #     logger.info(f"Generating lag data with {', '.join(lag_list)} lags")

    #     for lag in lag_list:
    #         ft_name = "item_cnt_shifted%s" % lag
    #         train_monthly[ft_name] = (
    #             train_monthly.sort_values("date_block_num")
    #             .groupby(["shop_id", "item_category_id", "item_id"])["item_cnt"]
    #             .shift(lag)
    #         )
    #         # Fill the empty shifted features with 0
    #         train_monthly[ft_name].fillna(0, inplace=True)
    #     train_monthly["item_trend"] = train_monthly["item_cnt"]

    #     for lag in lag_list:
    #         ft_name = "item_cnt_shifted%s" % lag
    #         train_monthly["item_trend"] -= train_monthly[ft_name]

    #     train_monthly["item_trend"] /= len(lag_list) + 1

    #     logger.info("End generating rolling data")
    #     return train_monthly

    def generate(self):
        logger.info("Start feature generator")

        train_data, test_data = self.get_datasets()
        monthly_data = self.clean_data(train_data, test_data)
        filled_data = self.fill_data(monthly_data)
        filled_data["year"] = filled_data["date_block_num"].apply(
            lambda x: ((x // 12) + 2013)
        )
        filled_data["month"] = filled_data["date_block_num"].apply(lambda x: (x % 12))

        ## assuring cleaning
        filled_data = filled_data.query(
            "item_cnt >= 0 and item_cnt <= 20 and item_price < 400000"
        )

        logger.info("Shifting targets with -1")

        filled_data["item_cnt_month"] = (
            filled_data.sort_values("date_block_num")
            .groupby(["shop_id", "item_id"])["item_cnt"]
            .shift(-1)
        )

        filled_data["item_price_unit"] = (
            filled_data["item_price"] // filled_data["item_cnt"]
        )
        filled_data["item_price_unit"].fillna(0, inplace=True)

        gp_item_price = (
            filled_data.sort_values("date_block_num")
            .groupby(["item_id"], as_index=False)
            .agg({"item_price": [np.min, np.max]})
        )
        gp_item_price.columns = [
            "item_id",
            "hist_min_item_price",
            "hist_max_item_price",
        ]

        filled_data = pd.merge(filled_data, gp_item_price, on="item_id", how="left")
        filled_data["price_increase"] = (
            filled_data["item_price"] - filled_data["hist_min_item_price"]
        )
        filled_data["price_decrease"] = (
            filled_data["hist_max_item_price"] - filled_data["item_price"]
        )

        # lagged_date = self.generate_rolling(filled_data)

        train_set = filled_data.query(
            "date_block_num >= 3 and date_block_num < 28"
        ).copy()
        validation_set = filled_data.query(
            "date_block_num >= 28 and date_block_num < 33"
        ).copy()

        test_set = filled_data.query("date_block_num == 33").copy()

        ## drop na
        train_set.dropna(subset=["item_cnt_month"], inplace=True)
        validation_set.dropna(subset=["item_cnt_month"], inplace=True)
        train_set.dropna(inplace=True)
        validation_set.dropna(inplace=True)

        X_train = train_set.drop(["item_cnt_month", "date_block_num"], axis=1)
        Y_train = train_set["item_cnt_month"].astype(int)
        X_validation = validation_set.drop(["item_cnt_month", "date_block_num"], axis=1)
        Y_validation = validation_set["item_cnt_month"].astype(int)

        ## reset data types
        int_features = ["shop_id", "item_id", "year", "month"]

        X_train[int_features] = X_train[int_features].astype("int32")
        X_validation[int_features] = X_validation[int_features].astype("int32")

        latest_records = pd.concat([train_set, validation_set]).drop_duplicates(
            subset=["shop_id", "item_id"], keep="last"
        )
        X_test = pd.merge(
            test_data,
            latest_records,
            on=["shop_id", "item_id"],
            how="left",
            suffixes=["", "_"],
        )
        X_test["year"] = 2015
        X_test["month"] = 9
        X_test.drop("item_cnt_month", axis=1, inplace=True)
        X_test[int_features] = X_test[int_features].astype("int32")

        X_test = X_test[X_train.columns]

        sets = [X_train, X_validation, X_test]

        # Replace missing values with the median of each shop.
        for dataset in sets:
            for shop_id in dataset["shop_id"].unique():
                for column in dataset.columns:
                    shop_median = dataset[(dataset["shop_id"] == shop_id)][
                        column
                    ].median()
                    dataset.loc[
                        (dataset[column].isnull()) & (dataset["shop_id"] == shop_id),
                        column,
                    ] = shop_median

        # Fill remaining missing values on test set with mean.
        X_test.fillna(X_test.mean(), inplace=True)

        X_train.drop(["item_category_id"], axis=1, inplace=True)
        X_validation.drop(["item_category_id"], axis=1, inplace=True)
        X_test.drop(["item_category_id"], axis=1, inplace=True)

        logger.info(f"Train set records: {train_set.shape[0]}")
        logger.info(f"Validation set records: {validation_set.shape[0]}")
        logger.info(f"Test set records: {test_set.shape[0]}")

        logger.info(
            "Train set records: %s (%.f%% of complete data)"
            % (train_set.shape[0], ((train_set.shape[0] / monthly_data.shape[0]) * 100))
        )
        logger.info(
            "Validation set records: %s (%.f%% of complete data)"
            % (
                validation_set.shape[0],
                ((validation_set.shape[0] / monthly_data.shape[0]) * 100),
            )
        )
        logger.info(X_train.head())

        return X_train, Y_train, X_validation, Y_validation, X_test


if __name__ == "__main__":

    fg = FeatureGenerator()
    X_train, Y_train, X_validation, Y_validation, X_test = fg.generate()
    # logger.info(monthly_data.head())
