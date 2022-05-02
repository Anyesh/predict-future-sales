import glob
import itertools as IT
import os
import tempfile
import urllib.request

import requests
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def no_files(path, file_list=[]):
    file_exist = [f for f in file_list if os.path.isfile(os.path.join(path, f))]
    return list(set(file_exist) ^ set(file_list))


def download_data(url, output_path):
    if not url:
        raise ValueError("Invalid URL to download data from")
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def uniquify(path, sep=""):
    def name_sequence():
        count = IT.count()
        yield ""
        while True:
            yield "{s}_{n:d}".format(s=sep, n=next(count))

    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


def get_latest_file(dir_path, file_type="*"):
    list_of_files = glob.glob(os.path.join(dir_path, file_type))
    return max(list_of_files, key=os.path.getctime)
