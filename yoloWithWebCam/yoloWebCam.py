import math

from ultralytics import YOLO
import cv2
import cvzone
import pyttsx3
speech=pyttsx3.init()
cap=cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,420)
cap.set(5,60)
model=YOLO("../yoloWeights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
prev_frame_time = 0
new_frame_time = 0

while True:
    success,img=cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,'',x2)
            if((x1<300)):
                print("left")
                speech.say("turn left")
                speech.runAndWait()
            elif(x1>300):
                print("right")
                speech.say("turn right")
                speech.runAndWait()
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            conf= math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))
            cls = int(box.cls[0])
            print(classNames[cls])
            speech.say(classNames[cls])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)




    cv2.imshow("Image",img)
    cv2.waitKey(1)
