from ultralytics import YOLO

class Yolotrain:
    def __init__(self,img):
        pass
        
    def TrainVal(data,source = 'model//yolov8n.pt',loop = '5') :
        model = YOLO(source)
        model.train(data=data , epochs=loop)
        model.val()
    
    def __del__(self):
        pass