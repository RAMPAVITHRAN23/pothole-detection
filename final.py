from ultralytics import YOLO
import cvzone
import cv2
import math
import serial
import time

# Running real-time from webcam
cap = cv2.VideoCapture(0)

ser =serial.Serial('COM7',9600)

model = YOLO('C:/Users/rpram/Music/pathole/best.pt')

# Reading the classes
classnames = ['pothole']
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)
    s1 = 0

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                scale=1.5, thickness=2)

                
                if classnames[Class] == "pothole":
                    ser.write(b'A')
                    print("pothole")

                    
                else:
                    print("Not pothole")
                    

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cvzone
# import cv2
# import math

# # Running real time from webcam
# cap = cv2.VideoCapture(0)

# model = YOLO('tomato_final.pt')

# # Reading the classes
# classnames = ['green','fully_ripened','half_ripened']
# while True:
#     ret,frame = cap.read()
#     #img =cv2.imread("2.jpg")
#     frame = cv2.resize(frame,(640,480))
#     result = model(frame,stream=True)

#     # Getting bbox,confidence and class names informations to work with
#     for info in result:
#         boxes = info.boxes
#         for box in boxes:
#             confidence = box.conf[0]
#             confidence = math.ceil(confidence * 100)
#             Class = int(box.cls[0])
#             if confidence > 50:
#                 x1,y1,x2,y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
#                 cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
#                                    scale=1.5,thickness=2)
#                 if classnames[Class] == "half ripened":
#                     print("half ripened")
#                 elif classnames[Class] == "green":
#                     print("half ripened")
#                 else:
#                     print("fully ripened")

#     cv2.imshow('frame',frame)
#     cv2.waitKey(1)