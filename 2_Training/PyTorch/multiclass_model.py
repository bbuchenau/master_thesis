import os
import csv
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from varname import nameof
import seaborn as sns
import numpy as np
import socket

# Import classes from my other files.
import data
import model
import visualization

# Set working directory and load trainingConfig JSON file that stores parameters.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
config_filepath = os.path.join(current_dir, "trainingConfig.json")
with open(config_filepath, "r") as jsonfile:
    config = json.load(jsonfile)

# Set dataset path. As I am working on different PCs, set path device-dependent.
pc_name = socket.gethostname()
if pc_name == "R184W10-CELSIUS":
    dataset_path = config["data_path_BICC"]
elif pc_name == "LAPTOP_UBUH5BJN":
    dataset_path = config["data_path_laptop"]
else:
    print("Wrong dataset path, check again.")

model_output_name = config["model_output_name"]

# Set transforms: data augmentation.
transform = transforms.Compose([
    transforms.Resize(list(config["transform_resize"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=config["rotation_degrees"]),
    transforms.ToTensor()
])

# Create the dataset.
dataset = data.MultiLabelImageDataset(root_dir=dataset_path, transform=transform)

# Split images into training and validation.
train_size = int(config["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the custom collate function (TODO: Find out how exactly this works!)
def custom_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

# Function to save training results to a csv file.
def results_to_csv(file_path, *lists):
    # Transpose the lists to create columns.
    columns = list(map(list, zip(*lists)))

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row.
        # TODO: Find correct way to use list variable names as string for header row. 
        # Until that, look as lists passed to function below.
        csv_writer.writerow(["epoch", "train/loss", "val/loss"])

        # Write data rows.
        for epoch, values in enumerate(columns, start=1):
            rounded_values = [round(value, 5) for value in values]
            csv_writer.writerow([epoch] + rounded_values)

# Save starting time.
start_time = time.time()

# Workaround, as there were runtime issues.
if __name__ == "__main__":
    # Create data loaders with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=custom_collate)

    # Instantiate the model
    num_classes = len(dataset.classes)
    model = model.CNN(num_classes)

    # Define loss function and optimizer: BCE loss function and Adam optimizer is good for multilabel classification problem.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])

    # Training loop
    num_epochs = config["epochs"] # For testing

    # Metrics for evaluation and visualisation.
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):

        all_labels = []  # Store all labels from validation set
        best_predictions = []  # Store best predictions from validation set

        model.train()
        epoch_train_losses = []
        epoch_train_predictions = []
        epoch_train_labels = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            predictions = torch.sigmoid(outputs).round().detach().cpu().numpy()
            epoch_train_predictions.extend(predictions)
            epoch_train_labels.extend(labels.cpu().numpy())

        # Calculate and log average training loss for the epoch
        average_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}")
        train_losses.append(average_loss)

        # Calculate training F1 score
        train_precision = precision_score(epoch_train_labels, epoch_train_predictions, average='micro')
        train_recall = recall_score(epoch_train_labels, epoch_train_predictions, average='micro')
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
        train_f1_scores.append(train_f1)

        # Validation loop within epoch.
        model.eval()
        epoch_val_losses = []
        epoch_val_predictions = []
        epoch_val_labels = []
        current_best_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.float())
                epoch_val_losses.append(val_loss.item())

                predictions = torch.sigmoid(outputs).round().detach().cpu().numpy()
                epoch_val_predictions.extend(predictions)
                epoch_val_labels.extend(labels.cpu().numpy())

                # Store labels and predictions for the confusion matrix
                all_labels.extend(labels.numpy())
                current_best_predictions.extend(torch.sigmoid(outputs).round().numpy())

        average_val_loss = sum(epoch_val_losses) / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}")
        val_losses.append(average_val_loss)

        # TODO: Check if this should be included in conditional below, and if correct matrix is calc!
        # Update best predictions based on the highest predicted probability for each class.
        if not best_predictions or val_loss < min(val_losses):
            best_predictions = current_best_predictions

        # Update best predictions based on the lowest val/loss.
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            # Save the model at the best epoch.
            torch.save(model.state_dict(), f"{model_output_name}_best.pth")
            # Print for testing.
            print("Model improved in epoch", epoch + 1)

        # Calculate validation f1 score
        val_precision = precision_score(epoch_val_labels, epoch_val_predictions, average='micro')
        val_recall = recall_score(epoch_val_labels, epoch_val_predictions, average='micro')
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        val_f1_scores.append(val_f1)

    # Print training duration.
    print("Training runtime:", int(time.time() - start_time), "seconds")

    # Save the trained model
    torch.save(model.state_dict(), f"{model_output_name}_last.pth")

    # Save results to csv file.
    results_to_csv('results.csv', train_losses, val_losses)

    # Visualize training and validation loss over epochs.
    visualization.visualize_losses(train_losses, "train_loss", "svg")
    visualization.visualize_losses(val_losses, "val_loss", "svg")

    # Visualize training and validation f1 score over epochs.
    visualization.visualize_losses(train_f1_scores, "train_f1", "svg")
    visualization.visualize_losses(val_f1_scores, "val_f1", "svg")

    # Create a confusion matrix based on the best predictions
    conf_matrix = multilabel_confusion_matrix(np.array(all_labels), np.array(best_predictions), labels=[0, 1, 2])
    class_names = dataset.classes

    # Normalize the confusion matrices
    normalized_conf_matrix = [conf_matrix[i] / conf_matrix[i].sum(axis=1, keepdims=True) for i in range(len(class_names))]
    
    # Create the multilabel confusion matrix using seaborn heatmap.
    for i in range(len(class_names)):
        visualization.visualize_conf_matrix(conf_matrix[i], class_names[i], "svg", "d", "abs")
        visualization.visualize_conf_matrix(normalized_conf_matrix[i], class_names[i], "svg", ".2f", "norm")

