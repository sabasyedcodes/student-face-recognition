import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from mtcnn import MTCNN

# Path to the folder containing captured images
data_path = "student_images"

# Initialize MTCNN face detector
detector = MTCNN()

# Load images and labels
images = []
labels = []
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        if results:
            x1, y1, w, h = results[0]["box"]
            x2, y2 = x1 + w, y1 + h
            face = img_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            images.append(face)
            labels.append(folder_name)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert data to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize image data
images = images.astype('float32') / 255.0

# Create model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save model weights
model.save('face_recognition_model.h5')

# Save label encoder
np.save('label_encoder.npy', label_encoder.classes_)
