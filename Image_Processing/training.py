from ultralytics import YOLO

# Train the model.
model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)
model.train(data="D:\\ben_masterthesis\OIDv4\Classification_binary", 
           epochs = 40)





