from PIL import Image
import os
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
from skimage.transform import resize

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Define the same CNN model like in the training script.
# TODO: Write to separate file and import to both training and prediction script.
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.base_model = resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load the model
model = CNN(3)
model.load_state_dict(torch.load("D:/ben_masterthesis/master_thesis/PyTorch/multiclass_test.pth"))
model.eval()

# Load and preprocess the image, now just for one to test it.
# TODO: Iterate over all test images.
image_path = "D:/ben_masterthesis/master_thesis/PyTorch/testing/tank_2.jpg"
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image = transform(image).unsqueeze(0)

# Predict image classes and save the output.
with torch.no_grad():
    output = model(input_image)
    predictions = torch.sigmoid(output)

# Print the predicted values for each class. Classes sorted alphabetically in training script, 
# so the indexing order is: 0 = soldier, 1 = tank, 2 = weapon. 
formatted_predictions = ["{:.4f}".format(value) for value in predictions.numpy().flatten()]
print("Predicted values for each class:")
print(formatted_predictions)

# HEATMAP VISUALIZATION

# Visualize last convolutional layer, directly before fully connected layer. TODO: Check if better options!
target_layer = model.base_model.layer4[-1].conv2

# Create a LayerGradCam object and generate heatmap.
cam = LayerGradCam(model, target_layer)
attributions = cam.attribute(input_image, target=2)

# Resize heatmap to 100x100 - I can adjust accuracy here.
resized_heatmap = resize(attributions.squeeze().cpu().detach().numpy(), (100, 100), mode='reflect')

# Normalize to be in range 0-1.
resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())

# Apply colormap with transparency directly to the heatmap.
rgba_heatmap = plt.get_cmap('viridis')(resized_heatmap)

# Convert the input_image tensor to a numpy array and normalize to range 0-1.
original_image_np = input_image.squeeze(0).cpu().numpy()
original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())

# Display the original image with the heatmap overlayed.
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(original_image_np.transpose((1, 2, 0)))
ax.imshow(rgba_heatmap, alpha=0.7, extent=(0, 224, 224, 0))
ax.axis('off')

# Show the plot
plt.savefig('graphics/activation_heatmap.svg', format='svg')
plt.show()
