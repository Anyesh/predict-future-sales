import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from . import settings
from .settings import logger
from .utils import download_data, no_files


class FeatureGenerator:
    def __init__(self):
        self.download_data()

    def download_data(self):
        if no_files(settings.DATA_DIR, settings.DATA_NAMES):
            for name in settings.DATA_NAMES:
                _ = download_data(f"{str(settings.DATA_URL)}/{str(name)}", settings.DATA_DIR)
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

        # logger.info("Joining items, shops and item_categories to sales")

        # train_data = (
        #     sales.join(items, on="item_id", rsuffix="_")
        #     .join(shops, on="shop_id", rsuffix="_")
        #     .join(item_categories, on="item_category_id", rsuffix="_")
        #     .drop(["item_id_", "shop_id_", "item_category_id_"], axis=1)
        # )

        return sales, items, shops, item_categories, test_data

    def clean_data(self, sales, items, shops, item_categories, test):
        logger.info("Start data cleaning")

        sales_items = pd.merge(sales, items, on="item_id", how="left")
        sales_items = sales_items.drop("item_name", axis=1)

        index_cols = ["shop_id", "item_id", "date_block_num"]

        data_grid = []
        for block_num in sales_items["date_block_num"].unique():
            cur_shops = sales_items.loc[
                sales_items["date_block_num"] == block_num, "shop_id"
            ].unique()
            cur_items = sales_items.loc[
                sales_items["date_block_num"] == block_num, "item_id"
            ].unique()
            data_grid.append(
                np.array(
                    list(product(*[cur_shops, cur_items, [block_num]])), dtype="int32"
                )
            )
        data_grid = pd.DataFrame(
            np.vstack(data_grid), columns=index_cols, dtype=np.int32
        )

        mean_sales = (
            sales_items.groupby(["date_block_num", "shop_id", "item_id"])
            .agg({"item_cnt_day": "sum", "item_price": np.mean})
            .reset_index()
        )
        mean_sales = pd.merge(
            data_grid,
            mean_sales,
            on=["date_block_num", "shop_id", "item_id"],
            how="left",
        ).fillna(0)

        mean_sales = pd.merge(mean_sales, items, on="item_id", how="left")

        ## Indiv encoding
        for type_id in ["item_id", "shop_id", "item_category_id"]:
            for column_id, aggregator, aggtype in [
                ("item_price", np.mean, "avg"),
                ("item_cnt_day", np.sum, "sum"),
                ("item_cnt_day", np.mean, "avg"),
            ]:

                mean_df = (
                    sales_items.groupby([type_id, "date_block_num"])
                    .aggregate(aggregator)
                    .reset_index()[[column_id, type_id, "date_block_num"]]
                )
                mean_df.columns = [
                    type_id + "_" + aggtype + "_" + column_id,
                    type_id,
                    "date_block_num",
                ]
                mean_sales = pd.merge(
                    mean_sales, mean_df, on=["date_block_num", type_id], how="left"
                )

        lag_variables = list(mean_sales.columns[7:]) + ["item_cnt_day"]

        lags = [1, 2, 3, 6]  # lag values

        for lag in tqdm(lags):

            sales_new_df = mean_sales.copy()
            sales_new_df.date_block_num += lag
            sales_new_df = sales_new_df[
                ["date_block_num", "shop_id", "item_id"] + lag_variables
            ]
            sales_new_df.columns = ["date_block_num", "shop_id", "item_id"] + [
                lag_feat + "_lag_" + str(lag) for lag_feat in lag_variables
            ]
            mean_sales = pd.merge(
                mean_sales,
                sales_new_df,
                on=["date_block_num", "shop_id", "item_id"],
                how="left",
            )

        ## using 2015 data
        mean_sales = mean_sales[mean_sales["date_block_num"] > 12]

        ## filling missing values
        for feat in mean_sales.columns:
            if "item_cnt" in feat:
                mean_sales[feat] = mean_sales[feat].fillna(0)
            elif "item_price" in feat:
                mean_sales[feat] = mean_sales[feat].fillna(mean_sales[feat].median())

        cols_to_drop = [x for x in lag_variables if x not in ["item_cnt_day"]] + [
            "item_price",
            "item_name",
        ]

        train_data = mean_sales.drop(cols_to_drop, axis=1)

        ## test setup
        test["date_block_num"] = 34
        test = pd.merge(test, items, on="item_id", how="left")

        for lag in tqdm(lags):

            sales_new_df = mean_sales.copy()
            sales_new_df.date_block_num += lag
            sales_new_df = sales_new_df[
                ["date_block_num", "shop_id", "item_id"] + lag_variables
            ]
            sales_new_df.columns = ["date_block_num", "shop_id", "item_id"] + [
                lag_feat + "_lag_" + str(lag) for lag_feat in lag_variables
            ]
            test = pd.merge(
                test,
                sales_new_df,
                on=["date_block_num", "shop_id", "item_id"],
                how="left",
            )
        ## TODO: add Assertion for uniform cols

        test = test.drop(["ID", "item_name"], axis=1)

        for feat in test.columns:
            if "item_cnt" in feat:
                test[feat] = test[feat].fillna(0)
            elif "item_price" in feat:
                test[feat] = test[feat].fillna(test[feat].median())

        logger.info("End data cleaning")

        return train_data, test

    def generate(self):
        logger.info("Start feature generator")

        sales, items, shops, item_categories, test_data = self.get_datasets()
        train, test = self.clean_data(sales, items, shops, item_categories, test_data)

        col = [c for c in train.columns if c not in ["item_cnt_day"]]
        train_data = train[train["date_block_num"] < 33]
        train_y = np.log1p(train_data["item_cnt_day"].clip(0, 20))
        train_data = train_data[col]

        val_data = train[train["date_block_num"] == 33]
        val_y = np.log1p(val_data["item_cnt_day"].clip(0, 20))
        val_data = val_data[col]

        return train_data, train_y, val_data, val_y, test, col


if __name__ == "__main__":

    fg = FeatureGenerator()
    train_data, train_y, val_data, val_y, test, cols = fg.generate()
    print(train_data.drop_duplicates())
    # logger.info(monthly_data.head())
