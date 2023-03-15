from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data="D:\_Uni\Master\Geoinformatics\WS2022\Camera_Trap_Challenge\\training\data.yaml", epochs = 1)