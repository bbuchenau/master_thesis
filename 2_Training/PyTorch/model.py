from torchvision.models import resnet18
import torch.nn as nn

finetune = True

# Define the CNN model. Here, I can modify the network.
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.base_model = resnet18(pretrained=True)

        # Use pretrained model only.
        if not finetune:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        # Use my finetuning settings.
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False

            in_features = self.base_model.fc.in_features

            # Remove the existing fully connected layer.
            self.base_model.fc = nn.Identity()

            # Add a new fully connected layer on top, where I can adjust size of dense layer and dropout.
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        if not finetune:
            return self.base_model(x)
        else:
            x = self.base_model(x)
            x = self.fc(x)
            return x