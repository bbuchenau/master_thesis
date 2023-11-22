import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function that creates my dataset, customized for local structure.
class MultiLabelImageDataset(Dataset):
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

# Define custom dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset.
dataset = MultiLabelImageDataset(root_dir="D:/ben_masterthesis/OIDv4_ToolKit/OID/Dataset_nl/train/TEST", transform=transform)

# Split the dataset into training and validation sets.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the custom collate function (Because...?)
def custom_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

# Workaround, as there were runtime issues.
if __name__ == "__main__":
    # Create data loaders with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # Define the CNN model. Here, I can modify the network.
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(CNN, self).__init__()
            self.base_model = resnet18(pretrained=True)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.base_model(x)

    # Instantiate the model
    num_classes = len(dataset.classes)
    model = CNN(num_classes)

    # Define loss function and optimizer: BCE loss function and Adam optimizer is good for multilabel classification problem.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2 # For testing
    train_losses = []
    val_losses = []
    all_labels = []  # Store all labels from validation set
    best_predictions = []  # Store best predictions from validation set

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

        # Validation loop within epoch.
        model.eval()
        total_val_loss = 0.0
        current_best_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.float())
                total_val_loss += val_loss.item()

                # Store labels and predictions for the confusion matrix
                all_labels.extend(labels.numpy())
                current_best_predictions.extend(torch.sigmoid(outputs).round().numpy())

        # Update best predictions based on the highest predicted probability for each class
        if not best_predictions or total_val_loss < min(val_losses):
            best_predictions = current_best_predictions

        # Calculate and log average validation loss for the epoch
        average_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")
        val_losses.append(average_val_loss)

    # Save the trained model
    torch.save(model.state_dict(), "multiclass_test.pth")

    # Visualize training and validation loss over epochs
    plt.plot(train_losses, label='Training Loss')
    plt.savefig('graphics/train_loss.svg', format='svg')
    plt.show()
    plt.plot(val_losses, label='Validation Loss')
    plt.savefig('graphics/val_loss.svg', format='svg')
    plt.show()
    
    #TODO: How does confusion matrix work for multilabel? Figure out and fix inconsistent input variable error!!

    #print(best_predictions)
    #print(all_labels)

    # Create a confusion matrix based on the best predictions
    #conf_matrix = multilabel_confusion_matrix(np.array(all_labels), np.array(best_predictions), labels=[0, 1, 2])
    #class_names = dataset.classes

    # Display the multilabel confusion matrix using seaborn heatmap
    #plt.figure(figsize=(10, 8))
    #for i in range(len(class_names)):
    #    sns.heatmap(conf_matrix[i], annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    #    plt.xlabel("Predicted")
    #    plt.ylabel("True")
    #    plt.show()


