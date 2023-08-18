import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name of the person")
parser.add_argument("roll_no", help="Roll number of the person")
args = parser.parse_args()

name = args.name
roll_no = args.roll_no

import cv2
import os
from mtcnn import MTCNN

# Create folder to store captured images
if not os.path.exists('student_images'):
    os.makedirs('student_images')

# Initialize MTCNN for face detection
detector = MTCNN()

# Create folder to store images for current student
folder_path = os.path.join('student_images', f'{name}_{roll_no}')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Capture images
num_images = 0
camera = cv2.VideoCapture(0)
while num_images < 100:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        continue
        
    # Detect face using MTCNN
    result = detector.detect_faces(frame)
    if len(result) > 0:
        # Save face region as image
        x, y, w, h = result[0]['box']
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(folder_path, f'{name}_{roll_no}_{num_images}.jpg'), face_img)
        num_images += 1
            
    cv2.imshow('Capture Images', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
camera.release()
cv2.destroyAllWindows()
