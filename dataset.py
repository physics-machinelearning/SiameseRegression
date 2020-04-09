# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
import os
from io import BytesIO
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class SiameseRegressionTrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        super(SiameseRegressionTrainDataset, self).__init__()
        self.x_train, self.y_train = x_train, y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        index1 = random.choice(range(len(self.x_train)))
        x_train0, y_train0 = self.x_train[index], self.y_train[index]
        x_train1, y_train1 = self.x_train[index1], self.y_train[index1]

        return (x_train0, y_train0), (x_train1, y_train1)


class SiameseRegressionTestDataset(Dataset):
    def __init__(self, x_test, y_test):
        super(SiameseRegressionTestDataset, self).__init__()
        self.x_test, self.y_test = x_test, y_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, index):
        index1 = random.choice(range(len(self.x_test)))
        x_test0, y_test0 = self.x_test[index], self.y_test[index]
        x_test1, y_test1 = self.x_test[index1], self.y_test[index1]

        return (x_test0, y_test0), (x_test1, y_test1)


class ConvRegressionTrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        super(ConvRegressionTrainDataset, self).__init__()
        self.x_train, self.y_train = x_train, y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x_train0, y_train0 = self.x_train[index], self.y_train[index]

        return x_train0, y_train0


class ConvRegressionTestDataset(Dataset):
    def __init__(self, x_test, y_test):
        super(ConvRegressionTestDataset, self).__init__()
        self.x_test, self.y_test = x_test, y_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, index):
        x_test0, y_test0 = self.x_test[index], self.y_test[index]

        return x_test0, y_test0


def _load(folderpath, datanum):
    imgs = []
    ages = []
    file_list = os.listdir(folderpath)
    random.seed(100)
    random_selected = random.sample(file_list, datanum)
    for filename in random_selected:
        age = int(filename[:filename.index('_')])
        ages.append(age)

        filepath = os.path.join(folderpath, filename)
        with open(filepath, 'rb') as f:
            binary = f.read()
        img = Image.open(BytesIO(binary))
        img_resize = img.resize((64, 64))
        img_array = np.asarray(img_resize)
        img_gray = 0.299 * img_array[:, :, 2] + 0.587 + img_array[:, :, 1] + 0.114 * img_array[:, :, 0]
        imgs.append(img_gray)

    ages = np.array(ages)
    imgs = np.array(imgs)

    x_train, x_test, y_train, y_test = train_test_split(imgs, ages, test_size=0.5, shuffle=True)

    x_train /= 255.
    x_test /= 255.

    x_train_tensor = torch.tensor(x_train)[:, None, :, :]
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test)[:, None, :, :]
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


if __name__ == "__main__":
    folderpath = 'SiameseRegression/bareface/crop_part1'
    x_train, y_train, x_test, y_test = _load(folderpath)
    print(x_test.shape)
    train_dataset = SiameseRegressionTrainDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = SiameseRegressionTrainDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    epochs = 100
    for epoch in range(epochs):
        i = 0
        for (x_train0, y_train0), (x_train1, y_train1) in test_dataloader:
            i += 1
        print(i)
        print(x_train0.shape, y_train0.shape)
        break