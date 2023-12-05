import os
from ultralytics import YOLO

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Train the model.
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model.train(data="D:\\ben_masterthesis\OIDv4\Classification_binary", 
           epochs = 1)





