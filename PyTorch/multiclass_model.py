import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MultiLabelImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = self._make_dataset()

    def _make_dataset(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    item = (img_path, [self.class_to_idx[class_name]])
                    images.append(item)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, labels = self.imgs[idx]

        # Read the image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert labels to a binary vector
        binary_label = torch.zeros(len(self.classes))
        binary_label[labels] = 1.0

        return image, binary_label

# Define your custom dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = MultiLabelImageFolderDataset(root_dir="D:/ben_masterthesis/OIDv4_ToolKit/OID/Dataset_nl/train/TEST", transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the custom collate function
def custom_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

if __name__ == "__main__":
    # Create data loaders with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # Define the CNN model
    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super(CustomCNN, self).__init__()
            self.base_model = resnet18(pretrained=True)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.base_model(x)

    # Instantiate the model
    num_classes = len(dataset.classes)
    model = CustomCNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Calculate and log average training loss for the epoch
        average_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}")
        train_losses.append(average_loss)

    # Validation loop
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            # Store labels and predictions for the confusion matrix
            all_labels.extend(labels.numpy())
            all_predictions.extend(torch.sigmoid(outputs).round().numpy())

        average_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {average_loss:.4f}")

    # Visualize training loss over epochs
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('graphics/train_loss.svg', format='svg')
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "multiclass_test.pth")
    
