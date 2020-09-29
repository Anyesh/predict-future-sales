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
            (train_data["item_price"] > 0 & train_data["item_price"] < 100000)
            & (train_data["item_cnt_day"] > 0 & train_data["item_cnt_day"] < 1001)
        ]

        logger.info("End data cleaning")

        return train_data

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

    def generate(self):
        logger.info("Start feature generator")

        train_data, test_data = self.get_datasets()
        clean_data = self.clean_data(train_data, test_data)
        filled_data = self.fill_data(clean_data)
        filled_data["year"] = filled_data["date_block_num"].apply(
            lambda x: ((x // 12) + 2013)
        )
        filled_data["month"] = filled_data["date_block_num"].apply(lambda x: (x % 12))

        ## PENDONG

        return X_train, Y_train, X_validation, Y_validation, X_test


if __name__ == "__main__":

    fg = FeatureGenerator()
    X_train, Y_train, X_validation, Y_validation, X_test = fg.generate()
    # logger.info(monthly_data.head())
