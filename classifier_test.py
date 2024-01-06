import urllib.request
import os
import pandas as pd
import numpy as np
from basic_knn import KNN

file_name = "abalone.data"


def prepare_dataset_file():
    # 文件下载链接
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

    # 检查文件是否存在
    if not os.path.isfile(file_name):
        # 如果文件不存在，则下载文件
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(url, file_name)
        print(f"{file_name} downloaded.")
    else:
        print(f"use cached {file_name}")


def prepare_dataset() -> tuple[np.ndarray, np.ndarray]:
    abalone = pd.read_csv(file_name, header=None)
    print(abalone.head())
    abalone.columns = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]
    abalone = abalone.drop("Sex", axis=1)
    X = abalone.drop("Rings", axis=1)
    X = X.values
    y = abalone["Rings"]
    # 研究鲍鱼环数和物理特征的关系。
    y = y.values
    return X, y


def main_perdict():
    # lode model
    prepare_dataset_file()
    X, y = prepare_dataset()
    model = KNN(X, y)

    new_data_point = np.array([
        0.569552,
        0.446407,
        0.154437,
        1.016849,
        0.439051,
        0.222526,
        0.291208,
    ])

    predict_result = model.predict(new_data_point)
    print(predict_result)


if __name__ == '__main__':
    main_perdict()
