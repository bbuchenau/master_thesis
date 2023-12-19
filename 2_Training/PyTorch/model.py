from torchvision.models import resnet50
import torch.nn as nn
import os
import json

# Set working directory and load trainingConfig JSON file that stores parameters.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
config_filepath = os.path.join(current_dir, "trainingConfig.json")
with open(config_filepath, "r") as jsonfile:
    config = json.load(jsonfile)

# Determine if model will be finetuned or not (currently used for testing).
finetune = config["finetuning"]

# Define the CNN model. Here, I can modify the network.
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.base_model = resnet50(pretrained=True)

        # Use pretrained model only.
        if not finetune:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        # Use my finetuning settings.
        else:
            # Freeze pretrained model layers.
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze the specified (last 3) layers to enable learning new features.
            # TODO: Analyse performance if not applied vs applied. 
            unfreeze_layer_names = config["unfreeze_layers"] # Specific to ResNet!
            for name, param in self.base_model.named_parameters():
                if any(unfreeze_layer_name in name for unfreeze_layer_name in unfreeze_layer_names):
                    param.requires_grad = True

            in_features = self.base_model.fc.in_features

            # Remove the existing fully connected layer.
            self.base_model.fc = nn.Identity()

            # Add a new fully connected layer on top, where I can adjust size of dense layer and dropout.
            self.fc = nn.Sequential(
                nn.Linear(in_features, config["dense_layer_nodes"]), 
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["dense_layer_nodes"], num_classes)
            )

    def forward(self, x):
        if not finetune:
            return self.base_model(x)
        else:
            x = self.base_model(x)
            x = self.fc(x)
            return x