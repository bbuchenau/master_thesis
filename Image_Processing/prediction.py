from ultralytics import YOLO
from PIL import Image
import os

# Import trained model.
model = YOLO("D:\\ben_masterthesis\\runs\classify\\train\weights\\last.pt")

# Run detection on directory.
image_directory = "D:\\ben_masterthesis\OIDv4\Classification\\test\\weapon"
file_list = os.listdir(image_directory)

# Iterate over all images in directory.
for image in file_list:
    results = model(os.path.join(image_directory, image))

    # Show results as PIL RGB image.
    for r in results:
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])
        im.show()