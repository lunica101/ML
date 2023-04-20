from ultralytics import YOLO
from func import Yolopred
from training import Yolotrain

#Yolotrain(data = "C:\\Users\\USER\\Desktop\\YoloML\\New\\data.yaml" ,source = "yolov8n.pt" ,loop = 10)
img = r'C:\Users\USER\Desktop\data\test\cat5.jpg'
Yolopred.getprediction(img)