import logging
import os

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)

logger = logging.getLogger()


BASEDIR = os.path.dirname(os.path.abspath("__file__"))

DATA_DIR = os.path.join(BASEDIR, "data")
CHK_PATH = os.path.join(BASEDIR, "checkpoints")

DATA_URL = "https://storage.googleapis.com/anyesh-data-bucket/sales-prediction"

DATA_NAMES = [
    "test.csv",
    "sales_train.csv",
    "items.csv",
    "item_categories.csv",
    "shops.csv",
]

OUTPUT_PATH = os.path.join(BASEDIR, "outputs")
