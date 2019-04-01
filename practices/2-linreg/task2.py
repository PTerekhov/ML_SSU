from glob import glob
import os
import random
import keras.applications.resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(42)

begin = "\n------BEGIN------\n"
end = "\n-------END-------\n"

print("Path to data: ")
path = input()

#path = "/Users/pterekhov/Projects/PycharmProjects/ML_SSU/data"
#print("Path to data:", path)


def read_dataset(path):
    X = []
    y = []

    image_paths_list = glob(os.path.join(path,'train', '*.jpg'))
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

X, y = read_dataset(path)
model = keras.applications.resnet50.ResNet50(weights='imagenet')
preds = model.predict(X)

print("preds: ", begin, preds, end)

preds_train, preds_test, y_train, y_test = train_test_split(preds, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial').fit(preds_train, y_train)
predictions = clf.predict(preds_test)

print("accuracy score: ", begin, accuracy_score(y_test, predictions), end)
print("confusion matrix: ", begin, confusion_matrix(y_test, predictions), end)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = [x.flatten() for x in X_train]
X_test = [x.flatten() for x in X_test]
clf = LogisticRegression(random_state=42,solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
predictions=clf.predict(X_test)

print("predictions: ", begin, predictions, end)
print("accuracy score: ", begin, accuracy_score(y_test, predictions), end)
print("confusion matrix: ", begin, confusion_matrix(y_test, predictions), end)

#/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras
#/Users/pterekhov/Projects/PycharmProjects/ML_SSU/data

