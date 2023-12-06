import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


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
