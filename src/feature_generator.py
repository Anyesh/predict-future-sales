import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

        sales["date"] = pd.to_datetime(sales["date"], format="%Y %m %d")
        sales["month"] = sales["date"].dt.month
        sales["year"] = sales["date"].dt.year

        sales = sales.drop("date", axis=1)
        sales = sales.drop("item_price", axis=1)

        temp = sales.groupby(
            [x for x in sales.columns if x not in ["item_cnt_day"]], as_index=False
        )["item_cnt_day"].sum()
        temp.columns = [
            "date_block_num",
            "shop_id",
            "item_id",
            "month",
            "year",
            "item_cnt_month",
        ]

        shop_item_mean = (
            temp[["shop_id", "item_id", "item_cnt_month"]]
            .groupby(["shop_id", "item_id"], as_index=False)["item_cnt_month"]
            .mean()
        )
        shop_item_mean.columns = ["shop_id", "item_id", "item_cnt_mean"]

        train = pd.merge(temp, shop_item_mean, how="left", on=["shop_id", "item_id"])

        shop_pre_month = train[train["date_block_num"] == 33][
            ["shop_id", "item_id", "item_cnt_month"]
        ]
        shop_pre_month.columns = ["shop_id", "item_id", "item_cnt_prev_month"]

        train = pd.merge(
            train, shop_pre_month, how="left", on=["shop_id", "item_id"]
        ).fillna(0)
        train = pd.merge(train, items, how="left", on=["item_id"])
        train = pd.merge(train, item_categories, how="left", on=["item_category_id"])
        train = pd.merge(train, shops, how="left", on=["shop_id"])
        train = train.drop_duplicates(subset=["shop_id", "item_id"], keep="last")

        test["month"] = 11
        test["year"] = 2015
        test["date_block_num"] = 34

        test = pd.merge(
            test, shop_item_mean, how="left", on=["shop_id", "item_id"]
        ).fillna(0)
        test = pd.merge(
            test, shop_pre_month, how="left", on=["shop_id", "item_id"]
        ).fillna(0)
        test = pd.merge(test, items, how="left", on=["item_id"]).fillna(0)
        test = pd.merge(
            test, item_categories, how="left", on=["item_category_id"]
        ).fillna(0)
        test = pd.merge(test, shops, how="left", on=["shop_id"]).fillna(0)
        test["item_cnt_month"] = 0
        test = test.drop_duplicates(subset=["shop_id", "item_id"], keep="last")

        for item in ["shop_name", "item_name", "item_category_name"]:
            lbl = LabelEncoder()
            lbl.fit(list(train[item].unique()) + list(test[item].unique()))
            train[item] = lbl.transform(train[item].astype(str))
            test[item] = lbl.transform(test[item].astype(str))

        logger.info("End data cleaning")

        return train, test

    def generate(self):
        logger.info("Start feature generator")

        sales, items, shops, item_categories, test_data = self.get_datasets()
        train, test = self.clean_data(sales, items, shops, item_categories, test_data)

        col = [c for c in train.columns if c not in ["item_cnt_month"]]
        train_data = train[train["date_block_num"] < 33]
        train_y = np.log1p(train_data["item_cnt_month"].clip(0, 20))
        train_data = train_data[col]

        val_data = train[train["date_block_num"] == 33]
        val_y = np.log1p(val_data["item_cnt_month"].clip(0, 20))
        val_data = val_data[col]

        ## PENDONG

        return train_data, train_y, val_data, val_y, test, col


if __name__ == "__main__":

    fg = FeatureGenerator()
    train_data, train_y, val_data, val_y, test, cols = fg.generate()
    print(train_data.drop_duplicates())
    # logger.info(monthly_data.head())
