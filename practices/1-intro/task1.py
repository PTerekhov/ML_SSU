import pandas as pd
import pathlib
import hashlib
import numpy as np
import random
from PIL import Image
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

path = "~/Projects/PycharmProjects/data/train/"
train_directory = pathlib.Path(path)
sample_size = 5000
STUDENT_ID = "PTerekhov-412"

def initialize_random_seed():
    """Инициализирует ГПСЧ из STUDENT_ID"""
    sha256 = hashlib.sha256()
    sha256.update(STUDENT_ID.encode("utf-8"))

    fingerprint = int(sha256.hexdigest(), 16) % (2 ** 32)

    random.seed(fingerprint)
    np.random.seed(fingerprint)


def read_target_variable():
    """Прочитаем разметку фотографий из названий файлов"""
    target_variable = {
        "filename": [],
        "is_cat": []
    }
    image_paths = list(train_directory.glob("*.jpg"))
    random.shuffle(image_paths)
    for image_path in image_paths[:sample_size]:
        filename = image_path.name
        class_name = filename.split(".")[0]
        target_variable["filename"].append(filename)
        target_variable["is_cat"].append(class_name == "cat")

    return pd.DataFrame(data=target_variable)

def read_data(target_df):
    """Читает данные изображений и строит их признаковое описание"""
    image_size = (100, 100)
    features = []
    target = []
    for i, image_name, is_cat in tqdm_notebook(target_df.itertuples(), total=len(target_df)):
        image_path = str(train_directory / image_name)
        image = Image.open(image_path)
        image = image.resize(image_size) # уменьшаем изображения
        image = image.convert('LA') # преобразуем в Ч\Б
        pixels = np.asarray(image)[:, :, 0]
        pixels = pixels.flatten()
        features.append(pixels)
        target.append(is_cat)
    return np.array(features), np.array(target)



initialize_random_seed()

target_df = read_target_variable()
print(target_df)

features, target = read_data(target_df)
print(features)

train, test = train_test_split(features, test_size=0.4)
valid = train_test_split(test, test_size=0.5)

model = linear_model.SGDClassifier(shuffle=True,max_iter=1000, tol=1e-3)
model.fit(train,test)