import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import torch.optim as optim
from torch import nn

from model import SiameseRegression, SiameseRegressionLoss


class TrainSiameseRegression:
    def __init__(self, model, train_dataloader, test_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = SiameseRegressionLoss()
        self.optimizer = optim.Adadelta(model.parameters())

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data0, data1) in enumerate(self.train_dataloader):
                inputs0, labels0 = data0
                inputs1, labels1 = data1

                self.optimizer.zero_grad()

                feature0, feature1, out0, out1 = self.model(inputs0, inputs1)
                loss = self.criterion(feature0, feature1, out0, out1, labels0, labels1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print('training loss = ', running_loss/len(self.train_dataloader))
            out_list, labels_list, feature_array = self.validation()

            if epoch % 10 == 0:
                self._plot(out_list, labels_list, feature_array)

    def validation(self):
        running_loss = 0.0
        out0_list = []
        out1_list = []
        labels0_list = []
        labels1_list = []
        for i, (data0, data1) in enumerate(self.test_dataloader):
            inputs0, labels0 = data0
            inputs1, labels1 = data1

            self.optimizer.zero_grad()

            feature0, feature1, out0, out1 = self.model(inputs0, inputs1)
            loss = self.criterion(feature0, feature1, out0, out1, labels0, labels1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            out0_list += out0.data.flatten().tolist()
            out1_list += out1.data.flatten().tolist()
            labels0_list += labels0.data.flatten().tolist()
            labels1_list += labels1.data.flatten().tolist()

            if i == 0:
                feature0_array = feature0.data
                feature1_array = feature1.data
            else:
                feature0_array = np.vstack((feature0_array, feature0.data))
                feature1_array = np.vstack((feature1_array, feature1.data))

        out_list = out0_list + out1_list
        labels_list = labels0_list + labels1_list
        feature_array = np.vstack((feature0_array, feature1_array))

        print('test loss = ', running_loss/len(self.test_dataloader))

        return out_list, labels_list, feature_array

    def _plot(self, out_list, labels_list, feature_array):
        plt.figure()
        plt.scatter(labels_list, out_list)
        plt.show()

        tsne = TSNE(n_components=2)
        feature_tsne = tsne.fit_transform(feature_array)
        
        plt.figure()
        plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=np.array(labels_list), cmap=cm.hsv)
        plt.colorbar()
        plt.show()


class TrainConvRegression:
    def __init__(self, model, train_dataloader, test_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adadelta(model.parameters())

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad()

                feature, out = self.model(inputs)
                out = out.float()
                labels = labels.float()
                loss = self.criterion(out, labels)
                loss = loss.float()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print('training loss = ', running_loss/len(self.train_dataloader))
            out_list, labels_list, feature_array = self.validation()

            if epoch % 10 == 0:
                self._plot(out_list, labels_list, feature_array)

    def validation(self):
        running_loss = 0.0
        out_list = []
        labels_list = []
        for i, (inputs, labels) in enumerate(self.test_dataloader):

            self.optimizer.zero_grad()

            feature, out = self.model(inputs)
            out = out.float()
            labels = labels.float()
            loss = self.criterion(out, labels)
            loss = loss.float()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            out_list += out.data.flatten().tolist()
            labels_list += labels.data.flatten().tolist()

            if i == 0:
                feature_array = feature.data
            else:
                feature_array = np.vstack((feature_array, feature.data))

        print('test loss = ', running_loss/len(self.test_dataloader))

        return out_list, labels_list, feature_array

    def _plot(self, out_list, labels_list, feature_array):
        plt.figure()
        plt.scatter(labels_list, out_list)
        plt.show()

        tsne = TSNE(n_components=2)
        feature_tsne = tsne.fit_transform(feature_array)
        
        plt.figure()
        plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=np.array(labels_list), cmap=cm.hsv)
        plt.colorbar()
        plt.show()



if __name__ == "__main__":
    from model import SiameseRegression, SiameseRegressionLoss
    from dataset import SiameseRegressionTestDataset, SiameseRegressionTrainDataset, _load
    from torch.utils.data import DataLoader

    model = SiameseRegression()
    criterion = SiameseRegressionLoss()
    
    folderpath = 'SiameseRegression/bareface/crop_part1'
    x_train, y_train, x_test, y_test = _load(folderpath)
    train_dataset = SiameseRegressionTrainDataset(x_train, y_train)
    test_dataset = SiameseRegressionTestDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    tsr = TrainSiameseRegression(model, train_dataloader, test_dataloader)
    tsr.train(100)