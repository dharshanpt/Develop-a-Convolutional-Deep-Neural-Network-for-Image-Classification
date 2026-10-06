# Developing a Neural Network Classification Model
# NAME: DHARSHAN P T
# REG NO: 212223230046
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import all the required libraries such as PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn, andSeaborn.

### STEP 2: 

Load the customer segmentation dataset from the CSV file and remove unnecessary columnssuch as the customer ID.

### STEP 3: 

Handle missing values in the dataset by replacing them with suitable values. Fill missing workexperience values with 0 and family size values with the median.

### STEP 4: 

Convert all categorical features into numerical values using Label Encoding so that they can beprocessed by the neural network.

### STEP 5: 

Encode the target column (Segmentation) into numerical classes and separate the dataset intoinput features (X) and target labels (Y).

### STEP 6: 

Split the dataset into training and testing sets. Apply feature scaling using StandardScaler tonormalize the input data.

### STEP 7: 

Convert the training and testing data into PyTorch tensors and create TensorDatasets andDataLoaders for batch processing.

### STEP 8: 

Define a feedforward neural network with four fully connected layers and ReLU activationfunctions.

### STEP 9: 

Initialize the model, define the Cross Entropy Loss function, and configure the Adam optimizer.

### STEP 10: 

Train the neural network for multiple epochs by performing forward propagation, losscalculation, backpropagation, and weight updates.

### STEP 11: 

Use the trained model to predict the segmentation classes for the test dataset.

### STEP 12: 

Evaluate the model using Accuracy Score, Confusion Matrix, and Classification Report.

### STEP 13: 

Visualize the confusion matrix using a Seaborn heatmap to analyze the classificationperformance of the model.


## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Check dataset
im, lbl = train_set[0]

print("Image shape:", im.shape)
print("Training images:", len(train_set))
print("Testing images:", len(test_set))

trl = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

tstl = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)

class CNNclassifier1(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.c2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.l1 = nn.Linear(
            128 * 3 * 3,
            64
        )

        self.l2 = nn.Linear(
            64,
            32
        )

        self.l3 = nn.Linear(
            32,
            10
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.c1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.l1(x))
        x = self.dropout(x)

        x = torch.relu(self.l2(x))

        x = self.l3(x)

        return x

model = CNNclassifier1()
model = model.to(device)

print(model)
criterion = nn.CrossEntropyLoss()

op = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
epochs = 5

for i in range(epochs):

    model.train()

    running_loss = 0.0

    for a, b in trl:

        # Move data to CPU/GPU
        a = a.to(device)
        b = b.to(device)

        # Clear previous gradients
        op.zero_grad()

        # Forward pass
        pred = model(a)

        # Calculate loss
        loss = criterion(pred, b)

        # Backward pass
        loss.backward()

        op.step()
        running_loss += loss.item()
    average_loss = running_loss / len(trl)
    print(
        f"Epoch [{i + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )
t = 0
c = 0
act = []
pre = []
model.eval()
with torch.no_grad():
    for img, labels in tstl:
        img = img.to(device)
        labels = labels.to(device)
        output = model(img)
        _, predicted = torch.max(output, 1)
        t += labels.size(0)
        c += (predicted == labels).sum().item()
        pre.extend(
            predicted.cpu().numpy()
        )
        act.extend(
            labels.cpu().numpy()
        )
accuracy = c / t * 100
print("\n--------------------------------")
print("Accuracy Score:", accuracy, "%")
print("--------------------------------")
conf_matrix = confusion_matrix(
    act,
    pre
)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(
    classification_report(
        act,
        pre,
        target_names=test_set.classes
    )
)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()
plt.show()
with torch.no_grad():
    img1, label = test_set[0]
    input_image = img1.unsqueeze(0).to(device)
    output = model(input_image)
    _, pred = torch.max(output, 1)
    classes = test_set.classes
    display_img = img1 * 0.3530 + 0.2860
    plt.figure(figsize=(4, 4))
    plt.imshow(
        display_img.squeeze(),
        cmap="gray"
    )
    plt.title(
        f"Predicted: {classes[pred.item()]}"
    )
    plt.axis("off")
    plt.show()
    print(
        f"Actual: {classes[label]}"
    )
    print(
        f"Predicted: {classes[pred.item()]}"
    )
original_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=None
)
image, label = original_dataset[0]
plt.figure(figsize=(4, 4))
plt.imshow(
    image,
    cmap="gray"
)
plt.title(
    original_dataset.classes[label]
)
plt.axis("off")
plt.show()
```


### Dataset Information


### OUTPUT

## Training Loss per Epoch:

<img width="260" height="125" alt="image" src="https://github.com/user-attachments/assets/02e6aa8f-2048-4244-ac41-6e6c647b19d3" />


## Confusion matrix:

<img width="988" height="871" alt="image" src="https://github.com/user-attachments/assets/be2c8d7a-5984-488d-b271-4268ba26a128" />
<img width="449" height="280" alt="image" src="https://github.com/user-attachments/assets/e668a9ce-ac23-4df3-ab67-0f55bdc480fa" />

## Classification report:

<img width="582" height="414" alt="image" src="https://github.com/user-attachments/assets/34937d3a-9ba6-4c7d-bfae-b9c43fa01d91" />


### New Sample Data Prediction

<img width="390" height="468" alt="image" src="https://github.com/user-attachments/assets/5b3c8d17-33c5-444a-adc4-935128d12a84" />


## RESULT
The Convolutional Neural Network was successfully developed and trained for image classification using PyTorch.


