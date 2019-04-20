from glob import glob
import os
import numpy as np
import random
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random.seed(42)

print("Path to data: ")
path = input()

#path = "/Users/pterekhov/Projects/PycharmProjects/ML_SSU/data"

def read_dataset(path):
    X = []
    y = []

    image_paths_list = glob(os.path.join(path, 'train', '*.jpg'))
    image_paths_sample = random.sample(image_paths_list, 100)

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

X, y = read_dataset(path)
model = ResNet50(weights='imagenet')
X.shape
preds = model.predict(X)
preds.shape

X_train, X_test, y_train, y_test = train_test_split(preds, y, test_size=0.33, random_state=42)

clf = XGBClassifier(reg_lambda=1)
FIT = clf.fit(X_train, y_train)

predictions_test = clf.predict(X_test)
accuracy = accuracy_score(y_test,predictions_test)
importance = clf.feature_importances_

print("fit: ", '\n', FIT, '\n')
print("predictions_test: ", '\n', predictions_test, '\n')
print("accuracy: ", '\n', accuracy, '\n')
print("importance: ", '\n', importance, '\n')
