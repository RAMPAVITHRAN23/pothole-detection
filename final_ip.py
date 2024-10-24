from ultralytics import YOLO
import cvzone
import cv2
import math
import threading
import time
import serial
# Define a class for threaded video capture
class VideoStream:
    def __init__(self, src=0):
        # Initialize the video stream from the IP camera URL or webcam
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread for updating frames
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep reading frames in a loop
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.stream.read()

    def read(self):
        # Return the current frame
        return self.frame

    def stop(self):
        # Stop the video stream
        self.stopped = True
        self.stream.release()

cap = VideoStream('http://192.168.1.8:8080/video').start()

# Load YOLO model
model = YOLO('C:/Users/rpram/Music/pathole/best.pt')
ser = serial.Serial('COM7',9600)
classnames = ['pothole']
frame_skip = 5
frame_count = 0

display_width = 480
display_height = 360

while True:
    # Capture frame from the video stream
    frame = cap.read()
    
    if frame is None:
        print("Failed to grab frame")
        break

    frame_count += 1
    
    # Process every Nth frame to reduce lag
    if frame_count % frame_skip == 0:
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Perform object detection on the frame
        result = model(frame, stream=True)

        # Iterate through the detection results
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)  # Convert to percentage
                Class = int(box.cls[0])

                # If confidence is above threshold, draw bounding box and label
                if confidence > 30:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%',[x1 + 8, y1 + 100], scale=1.5, thickness=2)
                    print(f'Object: {classnames[Class]}, Confidence: {confidence}%')

                    if classnames[Class] == "pothole":
                        print("pothole detected")
                        ser.write(b'A')
        # Resize the frame for display
        display_frame = cv2.resize(frame, (display_width, display_height))

        # Display the resulting frame
        cv2.imshow('frame', display_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.stop()
cv2.destroyAllWindows()
