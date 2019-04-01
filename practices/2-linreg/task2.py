import pandas as pd
from glob import glob
import os
import random

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(34)


def read_dataset(path):
    X = []
    y = []

    image_paths_list = glob(os.path.join('train', '*.jpg'))
    image_paths_sample = random.sample(image_paths_list, 1000)

    for image_path in image_paths_sample:
        image_name = os.path.basename(image_path)
        image_name_parts = image_name.split('.')
        label = image_name_parts[0] if len(image_name_parts) == 3 else None

        if label:
            y.append(int(label == 'cat'))

        x = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
        x = preprocess_input(x)

        X.append(x)

    return np.array(X), y


X, y = read_dataset('data')
model = ResNet50(weights='imagenet')
preds = model.predict(X)

preds_train, preds_test, y_train, y_test = train_test_split(
preds, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=42,solver = 'lbfgs', multi_class = 'multinomial').fit(preds_train, y_train)

predictions = clf.predict(preds_test)
accuracy_score(y_test, predictions)

confusion_matrix(y_test, predictions)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)
X_train = [x.flatten() for x in X_train]
X_test = [x.flatten() for x in X_test]

clf = LogisticRegression(random_state=42,solver = 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
print(predictions = clf.predict(X_test),  end='nnn')
print(accuracy_score(y_test, predictions), end='nnn')
print(confusion_matrix(y_test, predictions),  end='nnn')


