from ultralytics import YOLO

#TRAIN
model = YOLO('yolov8l-cls.pt')  # load a pretrained model (recommended for training)
model.train(data="D:\\ben_masterthesis\master_thesis\Image_Processing\gun_classification", epochs = 25)

#TEST
#model = YOLO("D:\\ben_masterthesis\master_thesis\\runs\classify\\train3\weights\\best.pt")
#results = model("D:\\ben_masterthesis\\backup_images\hekler koch usp", save = True)  # predict on an image


