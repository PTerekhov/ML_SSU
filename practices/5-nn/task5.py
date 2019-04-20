import subprocess

import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from google.colab import auth
from googleapiclient.discovery import build
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


def download_data(file_id, file_name):
    import io
    from googleapiclient.http import MediaIoBaseDownload

    request = drive_service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()

    downloaded.seek(0)
    with open(file_name, "wb") as f:
        f.write(downloaded.read())


auth.authenticate_user()
drive_service = build('drive', 'v3')

file_id = '139wA_Z9kustXy54ifhWWHJvARo5f7O6y'
file_name = 'star_wars.tar.gz'
download_data(file_id, file_name)

subprocess.run("tar xf star_wars.tar.gz", shell=True, check=True)

filenames = []
labels = []
for idx, class_dir in enumerate(os.listdir("star_wars")):
    print(f"берем файлы из папки \"{class_dir}\" и даем им класс {idx}")

    # не берем файлы кроме .jpg .jpeg и .png
    for file in os.listdir(os.path.join("star_wars", class_dir)):
        if not file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        filenames.append(os.path.join("star_wars", class_dir, file))
        labels.append(idx)


filenames[:10], labels[:10]

random_index = np.random.choice(range(len(filenames)))
test_img = cv2.imread(filenames[random_index])[:, :, ::-1]
print(labels[random_index])
plt.imshow(test_img)

train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels, test_size=0.33, random_state=42)

def add_pad(img, shape):
    color_pick = img[0][0]
    padded_img = color_pick * np.ones(shape + img.shape[2:3], dtype=np.uint8)
    x_offset = int((padded_img.shape[0] - img.shape[0]) / 2)
    y_offset = int((padded_img.shape[1] - img.shape[1]) / 2)
    padded_img[x_offset:x_offset + img.shape[0], y_offset:y_offset + img.shape[1]] = img

    return padded_img


def resize(img, shape):
    scale = min(shape[0] * 1.0 / img.shape[0], shape[1] * 1.0 / img.shape[1])
    if scale != 1:
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


# Задание 2
    def __init__(self, filenames, labels):
        self.filenames = filenames
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # мы должны отдать ему image в виде массива и соотвуствующий ему label
        filename = self.filenames[idx]
        label = self.label[idx]

        # мы получили имя файла, теперь нужно загрузить картинку как numpy array
        # и изменить размер так, чтобы он был 224 на 224
        img = cv2.imread(filename)

        img = resize(img, [224, 224])
        img = add_pad(img, (224, 224))

        # меняем порядок каналов и делим все на 255, оборачиваем в torch tensor
        # это просто надо делать, потом спросите зачем
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.
        return img, label


train_dataset = StarWarsDataset(train_filenames, train_labels)
train_dataloder = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=0)

# Задание 3
test_dataset = StarWarsDataset(test_filenames, test_labels)
test_dataloder = DataLoader(test_dataset, shuffle=True, batch_size=256, num_workers=0)

for batch in test_dataloder: # получаем 1 batch - 1 итерация подгрузки данных
  images, labels = batch     # наш Dataset возвращает tuple, поэтому мы можем сделать так
  print(f'Всего батчей по batch_size: {len(test_dataloder)}')
  print(f'Лейбл первого элемента в первом батче: {labels[0]}')
  print(f'Размер картинки в первом батче: {images[0].shape}')
  print(f'Картинка в первом батче: {images[0]}')
  break


model = resnet34(pretrained=True) # resnet обученный на ImageNet
for param in model.parameters():
  param.requires_grad=False


# loss и optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print(model)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.to('cuda')

def run_test_on_epoch(model, epoch, test_loader):
    model.eval()
    with torch.no_grad():
      test_accuracy = []
      test_real = []
      for batch_x, batch_y in tqdm(test_loader):
          outputs = model(batch_x.to('cuda')).detach().cpu().numpy()
          test_accuracy.append(outputs)
          test_real.append(batch_y.detach().cpu().numpy())
      print("Epoch", epoch, "test accuracy", accuracy_score(np.hstack(test_real), np.argmax(np.hstack(test_accuracy), axis=1)))
    model.train()


# Задание 5
for epoch in tqdm(range(25)):
    for batch in train_dataloder:
        images, labels = batch
        optimizer.zero_grad()
        output = model(images.to('cuda'))
        losse = criterion(output, labels.to('cuda'))
        losse.backward()
        optimizer.step()

    run_test_on_epoch(model, epoch, test_dataloder)


