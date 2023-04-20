from ultralytics import YOLO


class Yolopred:
    def __init__(self,img):
        pass
        
    def getprediction(img):
        model = YOLO("model//best.pt")
        predict = model.predict(source=img , save = True)
        return predict
    
    def __del__(self):
        pass