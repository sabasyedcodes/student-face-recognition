import cv2
import numpy as np
import os
import csv
from datetime import datetime
from datetime import date
import time

# Constants
FACE_SIZE = 160
TOLERANCE = 0.88

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load student images
student_images = {}
for student_name in os.listdir('student_images'):
    student_folder = os.path.join('student_images', student_name)
    if os.path.isdir(student_folder):
        student_images[student_name] = []
        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            student_images[student_name].append(img_path)

# Initialize attendance record
attendance = {}
for student_name in student_images.keys():
    attendance[student_name] = []

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Calculate the normalized histogram for the face
        hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist)

        # Compare the face histogram to each student's histograms
        matched = False
        for student_name, student_image_paths in student_images.items():
            for image_path in student_image_paths:
                img = cv2.imread(image_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                img_hist = cv2.normalize(img_hist, img_hist)

                score = cv2.compareHist(hist, img_hist, cv2.HISTCMP_CORREL)

                
                # If the score is above the threshold, mark the student as present
                if score >= TOLERANCE :
                    #attendance[student_name].append('present')

                    # Draw a green rectangle around the face
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw the student's name next to the rectangle
                    text = f'{student_name}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    text_x = x
                    text_y = y - 10
                    text_w = text_size[0]
                    text_h = text_size[1]
                    cv2.rectangle(frame, (text_x, text_y), (text_x + text_w, text_y + text_h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, text, (x, y-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    matched = True
                    break
            
            if matched:
                break
        else:
            # If no match is found, mark the face as unknown
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw "Unknown" next to the rectangle
            text = "Unknown"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x
            text_y = y - 10
            text_w = text_size[0]
            text_h = text_size[1]
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_w, text_y + text_h), color, cv2.FILLED)
            cv2.putText(frame, text,(x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
 # Show the frame with the detected faces
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Write attendance record to CSV file
with open('attendance.csv', 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Attendance','date','time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for student_name, record in attendance.items():
        writer.writerow({'Name':student_name, 'Attendance': len(record),'date':date.today(),'time': time.ctime(1627908313.717886)})

# Print the attendance record
print(attendance)   

