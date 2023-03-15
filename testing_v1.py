from ultralytics import YOLO

path = "D:\_Uni\Master\Geoinformatics\WS2022\Camera_Trap_Challenge\construction_test.jpeg"

model = YOLO("D:\_Uni\Master\Geoinformatics\\thesis\\runs\detect\\train4\weights\\last.pt") # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 is for webcam.
results = model.predict(source=path, show=True)

print(*results)