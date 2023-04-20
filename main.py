#import cv2
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# from PIL import Image
# import numpy as np
# from matplotlib.pyplot import 


model = YOLO("model//best.pt") #path to weights
img_path = "C:\\Users\\USER\\Desktop\\data\\test\\cat3.jpg"
predict = model.predict(source=img_path , save = True)
# model.export(format='onnx', dynamic=True)
# print(predict)
# print(f'{dir(model) = }')


#args = {}
#predictor = DetectionPredictor(overrides=args)
#predictor.predict_cli()
# kwargs, args
# kwargs = key word arguments
# args = arguments