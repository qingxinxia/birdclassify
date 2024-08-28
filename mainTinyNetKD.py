
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
from models import TinyNet, Net
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):

        soft_labels_teacher = F.softmax(teacher_logits / temperature, dim=1)


        distillation_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                                     soft_labels_teacher, reduction='batchmean') * (temperature ** 2)


        hard_loss = F.cross_entropy(student_logits, labels)


        total_loss = alpha * hard_loss + (1 - alpha) * distillation_loss
        return total_loss


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
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (1, 45, 20) -> (1, 1, 45, 20)
        x = self.resnet(x)
        return x

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
    # root_path = r'dataset/wmwb'
    root_path = r'D:\code\BirdMLClassification\transfer_learning'
    teacherWeightPath = "pretrain_ResNet18_49.pt"
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    teacher_model = ResNetClassifier(num_classes=20).to(device)
    # teacher_model.load_state_dict(torch.load(teacherWeightPath))
    # Load the saved checkpoint
    checkpoint = torch.load(teacherWeightPath)
    # checkpoint = torch.load(teacherWeightPath, map_location="mps")
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    # model = TinyNet(224, 20).to(device)
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 6: Training loop
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            #get teacher results:
            with torch.no_grad():
                outputsT = teacher_model(inputs)

            loss = distillation_loss(outputs, outputsT, labels, temperature=7, alpha=0.25)
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Step 7: Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # print("Training complete.")

        # Step 8: test loop
        model.eval()
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

    return

if __name__ == '__main__':
    main()