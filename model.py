import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from ultralytics import YOLO


model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")


model.train(data="C:/Users/SPIRO-PYTHON1/Desktop/bioooo_YOLO/data.yaml", epochs=10) 
