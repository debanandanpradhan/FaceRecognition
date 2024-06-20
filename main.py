import cv2 as cv
import numpy as np
import os
from mtcnn import MTCNN
from threading import Thread
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import pandas as pd
from datetime import datetime
import imutils
import time

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize FaceNet
facenet = FaceNet()

# Load face embeddings and encoder
faces_embeddings = np.load("groupfaces_embeddings_done_5classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model
model = pickle.load(open("svm_modelgroup_160x160.pkl", 'rb'))

# Initialize Excel file if it doesn't exist
excel_file = "face_recognition_log.xlsx"
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["Date", "Time", "Face Recognized"])
    df.to_excel(excel_file, index=False)
else:
    df = pd.read_excel(excel_file)

# Initialize set to store recognized users for each day
recognized_users_today = set()

# Set the confidence threshold (adjust as needed)
confidence_threshold = 0.4

# Function to perform face detection and recognition
def detect_faces(frame):
    global df  # Declare df as a global variable to modify it in the global scope
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Use MTCNN for face detection
    detections = detector.detect_faces(rgb_img)

    for detection in detections:
        x, y, w, h = detection['box']
        x, y = max(x, 0), max(y, 0)

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)

        # Extract face from the frame
        face_img = rgb_img[y:y + h, x:x + w]
        face_img = cv.resize(face_img, (160, 160))  # Resize face to match FaceNet input size
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        # Get embeddings of the face using FaceNet
        embeddings = facenet.embeddings(face_img)

        # Predict the class and confidence scores
        confidence_scores = model.predict_proba(embeddings)
        predicted_class = model.predict(embeddings)

        # Get the maximum confidence score and its index
        max_confidence_index = np.argmax(confidence_scores)
        max_confidence = confidence_scores[0, max_confidence_index]

        # Check if the maximum confidence score exceeds the threshold
        if max_confidence > confidence_threshold:
            final_name = encoder.inverse_transform(predicted_class)[0]
            cv.putText(frame, f"{final_name} ({max_confidence:.2f})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 3, cv.LINE_AA)

            # Get current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check if the user has already been recognized today
            if final_name not in recognized_users_today:
                # Append entry to DataFrame
                new_entry = pd.DataFrame(
                    {"Date": [current_date], "Time": [current_time], "Face Recognized": [final_name]})
                # Check if an entry for the user already exists for today
                if df[(df['Date'] == current_date) & (df['Face Recognized'] == final_name)].empty:
                    df = pd.concat([df, new_entry], ignore_index=True)
                else:
                    # Update the existing entry with the new time
                    df.loc[(df['Date'] == current_date) & (df['Face Recognized'] == final_name), 'Time'] = current_time
                recognized_users_today.add(final_name)  # Add user to recognized set for today
        else:
            # If confidence score is below threshold, mark as unknown
            cv.putText(frame, f"Unknown (Confidence: {max_confidence:.2f})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)

    # Write DataFrame to Excel file
    df.to_excel(excel_file, index=False)

# Function to continuously capture frames from the camera
def camera_stream():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)  # Resize frame for smoother processing

        # Create a thread to perform face detection
        thread = Thread(target=detect_faces, args=(frame,))
        thread.start()
        thread.join()  # Wait for thread to finish before processing next frame

        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Start camera streaming
    camera_stream()
