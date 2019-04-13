import pandas as pd
import pathlib
import hashlib
import numpy as np
import random
from PIL import Image
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook

print("Path to data: ")
path = input()

begin = "\n------BEGIN------\n"
end = "\n-------END-------\n"

#path = "/Users/pterekhov/Projects/PycharmProjects/ML_SSU/data/train"
#print("Path to data:", path)

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
print(target_df, '\n')

features, target = read_data(target_df)
print(features, '\n')

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.4)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5)


model = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal').fit(X_train, y_train)

predictions1 = model.predict(X_val)
predictions1 = [prediction >= 0.5 for prediction in predictions1]

predictions2 = model.predict(X_test)
predictions2 = [prediction >= 0.5 for prediction in predictions2]

predictions3 = model.predict(X_train)
predictions3 = [prediction >= 0.5 for prediction in predictions2]

accuracy1 = accuracy_score(y_val, predictions1)
accuracy2 = accuracy_score(y_test, predictions2)
accuracy3 = accuracy_score(y_train, predictions3)

print("accuracy1: ", accuracy1, '\n')
print("accuracy2: ", accuracy2, '\n')
print("accuracy3: ", accuracy3, '\n')

scaler = StandardScaler()

new_train = scaler.fit_transform(X_train)
new_val = scaler.fit_transform(X_val)
new_test = scaler.transform(X_test)

new_model = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal').fit(new_train, y_train)

new_predictions = new_model.predict(new_test)
new_predictions = [prediction >= 0.5 for prediction in new_predictions]
new_accuracy = accuracy_score(y_test, new_predictions)

print("new_accuracy: ", new_accuracy, '\n')