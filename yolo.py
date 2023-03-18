from ultralytics import YOLO
import cv2
model=YOLO('../yoloWeights/yolov8l.pt')
results=model("img/IMG_4390.JPG",show=True,save=True,save_txt=True)

cv2.waitKey(0)