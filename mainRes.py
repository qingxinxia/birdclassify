
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data.prepare_wmwb import list_subfolders, sliding_window_split
from models import *

class CustomDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# ResNet model


def load_data(datap):
    # datap = os.getcwd()
    with open(os.path.join(datap, 'train_wnwb.pkl'), 'rb') as f:
        [trainX, trainY0] = pickle.load(f)
    with open(os.path.join(datap, 'val_wnwb.pkl'), 'rb') as f:
        [valX, valY0] = pickle.load(f)
    with open(os.path.join(datap, 'test_wnwb.pkl'), 'rb') as f:
        [testX, testY0] = pickle.load(f)

    return trainX, trainY0, valX, valY0, testX, testY0

def main():
    root_path = r'D:\code\BirdMLClassification\transfer_learning'

    trainX, trainY0, valX, valY0, testX, testY0 = load_data(root_path)

    unique_labels = np.unique(trainY0)
    folderIDs = np.arange(len(unique_labels))
    label_dict = dict(zip(unique_labels, folderIDs))

    trainYIDs = [label_dict[i] for i in trainY0]
    valYIDs = [label_dict[i] for i in valY0]
    testYIDs = [label_dict[i] for i in testY0]


    # Converting data to PyTorch tensors
    batch_trainX = torch.tensor(trainX[:,:,:,0], dtype=torch.float32)
    batch_trainy = torch.tensor(trainYIDs, dtype=torch.long)

    batch_valX = torch.tensor(valX[:,:,:,0], dtype=torch.float32)
    batch_valy = torch.tensor(valYIDs, dtype=torch.long)

    batch_testX = torch.tensor(testX[:,:,:,0], dtype=torch.float32)
    batch_testy = torch.tensor(testYIDs, dtype=torch.long)


    # Step 3: Define DataLoader
    train_dataset = CustomDataset(batch_trainX, batch_trainy)
    val_dataset = CustomDataset(batch_valX, batch_valy)
    test_dataset = CustomDataset(batch_testX, batch_testy)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    # Step 5: Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Load the saved checkpoint
    # teacherWeightPath = "pretrain_ResNet18_40.pt"
    # checkpoint = torch.load(teacherWeightPath)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Step 6: Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # # Step 7: Validation loop
        # model.eval()
        # val_loss = 0.0
        # correct = 0
        # total = 0
        #
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


        if (epoch == num_epochs-1) or (epoch % 10 == 0):
        # if (epoch % 100 == 0) or (epoch == args.n_epoch-1):
            model_dir = 'pretrain_ResNet18_' + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            # torch.save(model.state_dict(), model_dir)
            torch.save({'model_state_dict': model.state_dict()}, model_dir)

        # Step 8: test loop
        # model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss / len(val_loader):.4f}, "
              f"Accuracy: {100 * correct / total:.2f}%")
    print("Training complete.")

    return

if __name__ == '__main__':
    main()