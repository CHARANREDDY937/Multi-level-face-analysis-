from flask import Flask, jsonify, Response
from flask_cors import CORS
import threading
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

app = Flask(__name__)
CORS(app)

# Initialize global variables
streaming_active = False
lock = threading.Lock()

# Load model once globally
model = None
face_dict = {0: "Akshaya", 1: "Charan", 2: "Charansai", 3: "Nida", 4: "Pranav", 5: "Risheel"}


def initialize_model():
    global model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    model.load_weights('person_classifier.h5')


# Initialize the model before the app starts
initialize_model()


# Function to generate frames
def generate_frames():
    global streaming_active, model, face_dict

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        streaming_active = False
        return

    try:
        while True:
            with lock:
                if not streaming_active:
                    cap.release()
                    break

            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, face_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        cap.release()


# Flask routes
@app.route('/video_feed')
def video_feed():
    global streaming_active
    with lock:
        if not streaming_active:
            return "Stream not started.", 403
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognize', methods=['POST'])
def recognize():
    global streaming_active
    with lock:
        if not streaming_active:
            streaming_active = True
    return jsonify({"status": "Recognition started successfully."}), 200


@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global streaming_active
    with lock:
        streaming_active = False
    return jsonify({"status": "Recognition stopped."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)


# # from flask import Flask, request, jsonify, Response
# # import os
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.utils import to_categorical
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# # from tensorflow.keras.models import Sequential
# # import cv2
# # from datetime import datetime
# # from csv import reader, writer

# # app = Flask(__name__)

# # # Helper functions
# # def load_training_data():
# #     images = []
# #     labels = []
# #     for file in os.listdir("data/"):
# #         if file.endswith("_images.npy"):
# #             student_id = int(file.split("_")[1])
# #             student_images = np.load(os.path.join("data", file))
# #             images.extend(student_images)
# #             labels.extend([student_id] * len(student_images))
# #     return np.array(images), np.array(labels)

# # def load_students():
# #     students = {}
# #     if os.path.exists("students.csv"):
# #         with open("students.csv", 'r') as file:
# #             csv_reader = reader(file)
# #             next(csv_reader)  # Skip header
# #             for row in csv_reader:
# #                 students[int(row[0])] = row[1]  # Map StudentID to Name
# #     return students

# # def update_attendance(student_name, attendance_file):
# #     current_date = datetime.now().date()

# #     # Read existing attendance
# #     if os.path.exists(attendance_file):
# #         with open(attendance_file, 'r') as file:
# #             csv_reader = reader(file)
# #             for row in csv_reader:
# #                 if row[0] == student_name and row[1] == str(current_date):
# #                     return False  # Already marked attendance for today

# #     # Append new attendance entry
# #     current_time = datetime.now()
# #     with open(attendance_file, 'a', newline='') as file:
# #         csv_writer = writer(file)
# #         csv_writer.writerow([student_name, current_date, current_time.time()])
# #     return True  # Successfully marked attendance

# # # Route to stream video feed
# # @app.route('/video_feed')
# # def video_feed():
# #     def generate_frames():
# #         cap = cv2.VideoCapture(0)  # Open the video capture device (webcam)
# #         while True:
# #             success, frame = cap.read()  # Read a frame
# #             if not success:
# #                 break
# #             else:
# #                 _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
# #                 frame = buffer.tobytes()  # Convert the frame to bytes
# #                 yield (b'--frame\r\n'
# #                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Send the frame as a multipart response
# #         cap.release()  # Release the video capture object
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # # Route to train the model
# # @app.route('/train_model', methods=['POST'])
# # def train_model():
# #     try:
# #         images, labels = load_training_data()
# #         images = images.astype('float32') / 255.0
# #         images = np.expand_dims(images, axis=-1)

# #         unique_labels = np.unique(labels)
# #         label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
# #         index_to_label = {idx: label for label, idx in label_to_index.items()}
# #         labels = np.array([label_to_index[label] for label in labels])
# #         labels = to_categorical(labels, num_classes=len(unique_labels))

# #         np.save("model/labels.npy", index_to_label)

# #         # Train the model
# #         datagen = ImageDataGenerator(
# #             rotation_range=20,
# #             width_shift_range=0.2,
# #             height_shift_range=0.2,
# #             horizontal_flip=True,
# #             zoom_range=0.2,
# #             shear_range=0.15
# #         )
# #         datagen.fit(images)

# #         model = Sequential([ 
# #             Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
# #             MaxPooling2D((2, 2)),
# #             Conv2D(128, (3, 3), activation='relu'),
# #             MaxPooling2D((2, 2)),
# #             Flatten(),
# #             Dense(256, activation='relu'),
# #             Dense(len(unique_labels), activation='softmax')
# #         ])
# #         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #         model.fit(datagen.flow(images, labels, batch_size=32), epochs=50)

# #         os.makedirs("model", exist_ok=True)
# #         model.save("model/cnn_model.h5")
# #         return jsonify({"message": "Model trained successfully."}), 200
# #     except Exception as e:
# #         return jsonify({"message": f"Failed to train model: {str(e)}"}), 500

# # # Route to add a student
# # @app.route('/add_student', methods=['POST'])
# # def add_student():
# #     data = request.json
# #     student_id = int(data.get('student_id'))
# #     name = data.get('name')
# #     department = data.get('department')

# #     if os.path.exists("students.csv"):
# #         with open("students.csv", 'r') as file:
# #             csv_reader = reader(file)
# #             next(csv_reader)  # Skip header
# #             for row in csv_reader:
# #                 if int(row[0]) == student_id:
# #                     return jsonify({"message": f"Student ID {student_id} already exists."}), 400

# #     if not os.path.exists("students.csv"):
# #         with open("students.csv", 'w', newline='') as file:
# #             csv_writer = writer(file)
# #             csv_writer.writerow(["StudentID", "Name", "Department"])
    
# #     with open("students.csv", 'a', newline='') as file:
# #         csv_writer = writer(file)
# #         csv_writer.writerow([student_id, name, department])

# #     # Capture images
# #     captured_images = []
# #     cap = cv2.VideoCapture(0)
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #     while len(captured_images) < 200:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# #         for (x, y, w, h) in faces:
# #             face = gray[y:y+h, x:x+w]
# #             face_resized = cv2.resize(face, (64, 64))
# #             captured_images.append(face_resized)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #     cap.release()
# #     cv2.destroyAllWindows()

# #     if len(captured_images) > 0:
# #         data_path = f"data/student_{student_id}_images.npy"
# #         np.save(data_path, np.array(captured_images))
# #         return jsonify({"message": f"Student {name} added successfully."}), 200
# #     else:
# #         return jsonify({"message": "Failed to capture images for the student."}), 500

# # # Route to mark attendance
# # # Global variable to control attendance marking
# # attendance_active = False

# # @app.route('/mark_attendance', methods=['GET'])
# # def mark_attendance():
# #     global attendance_active
# #     attendance_active = True  # Enable attendance marking

# #     try:
# #         model = load_model("model/cnn_model.h5")
# #         labels = np.load("model/labels.npy", allow_pickle=True).item()
# #         labels_reverse = {v: k for k, v in labels.items()}
# #         students = load_students()

# #         attendance_file = "attendance.csv"
# #         if not os.path.exists(attendance_file):
# #             with open(attendance_file, 'w', newline='') as file:
# #                 writer(file).writerow(["Name", "Date", "Time"])

# #         cap = cv2.VideoCapture(0)
# #         while attendance_active:  # Check the attendance_active flag
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# #             for (x, y, w, h) in faces:
# #                 face = gray[y:y+h, x:x+w]
# #                 face_resized = cv2.resize(face, (64, 64))
# #                 face_resized = face_resized.astype('float32') / 255.0
# #                 face_resized = np.expand_dims(face_resized, axis=-1)
# #                 face_resized = np.expand_dims(face_resized, axis=0)
# #                 predictions = model.predict(face_resized)
# #                 predicted_label = np.argmax(predictions)
# #                 student_id = labels_reverse.get(predicted_label, "Unknown")
# #                 student_name = students.get(student_id, "Unknown")
# #                 if student_name != "Unknown":
# #                     if update_attendance(student_name, attendance_file):
# #                         print(f"Marked attendance for {student_name}")
# #                     else:
# #                         print(f"{student_name} already marked for today.")
# #             if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow manual stop with 'q'
# #                 break
# #         cap.release()
# #         cv2.destroyAllWindows()
# #         return jsonify({"message": "Attendance marking stopped."}), 200
# #     except Exception as e:
# #         return jsonify({"message": f"Failed to mark attendance: {str(e)}"}), 500

# # @app.route('/stop_marking_attendance', methods=['GET'])
# # def stop_marking_attendance():
# #     global attendance_active
# #     attendance_active = False  # Disable attendance marking
# #     return jsonify({"message": "Attendance stopped successfully."}), 200



# # if __name__ == '__main__':
# #     app.run(debug=True, port=5020)
# import os
# import cv2
# import numpy as np
# import pandas as pd
# import datetime
# import tensorflow as tf
# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Initialize Flask app
# app = Flask(__name__)  # Fixed the incorrect syntax '_name'

# # File paths
# STUDENT_FILE = 'students.csv'
# ATTENDANCE_FILE = 'attendance.csv'
# MODEL_FILE = 'face_recognition_model.h5'
# LABEL_ENCODER_FILE = 'label_encoder.npy'
# DATASET_DIR = 'student_images'

# # Ensure necessary directories and files exist
# os.makedirs(DATASET_DIR, exist_ok=True)
# if not os.path.exists(STUDENT_FILE):
#     pd.DataFrame(columns=['ID', 'Name']).to_csv(STUDENT_FILE, index=False)
# if not os.path.exists(ATTENDANCE_FILE):
#     pd.DataFrame(columns=['Date', 'Time', 'ID', 'Name']).to_csv(ATTENDANCE_FILE)

# @app.route('/')
# def index():
#     """Render the main HTML page."""
#     return render_template('index.html')


# @app.route('/add_student', methods=['POST'])
# def add_student():
#     try:
#         data = request.get_json()
#         student_id = data.get('student_id')
#         student_name = data.get('student_name')

#         if not student_id or not student_name:
#             return jsonify({"error": "Missing student_id or student_name"}), 400

#         students = pd.read_csv(STUDENT_FILE)
#         if student_id in students['ID'].values:
#             return jsonify({"error": "Student ID already exists!"}), 400

#         new_student = pd.DataFrame({'ID': [student_id], 'Name': [student_name]})
#         students = pd.concat([students, new_student], ignore_index=True)
#         students.to_csv(STUDENT_FILE, index=False)

#         return jsonify({"message": "Student added successfully!"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/delete_student', methods=['POST'])
# def delete_student():
#     try:
#         data = request.get_json()
#         student_id = data.get('student_id')
#         if not student_id:
#             return jsonify({"error": "Missing student_id"}), 400

#         students = pd.read_csv(STUDENT_FILE)
#         if student_id not in students['ID'].values:
#             return jsonify({"error": "Student ID not found!"}), 404

#         students = students[students['ID'] != student_id]
#         students.to_csv(STUDENT_FILE, index=False)

#         return jsonify({"message": "Student deleted successfully!"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/capture_images', methods=['POST'])
# def capture_images():
#     try:
#         data = request.get_json()
#         student_id = data.get('student_id')
#         student_name = data.get('student_name')
#         if not student_id or not student_name:
#             return jsonify({"error": "Missing student_id or student_name"}), 400

#         num_images = 50  # Default number of images
#         student_dir = os.path.join(DATASET_DIR, f"{student_id}_{student_name}")
#         os.makedirs(student_dir, exist_ok=True)

#         cap = cv2.VideoCapture(0)
#         count = 0
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         while count < num_images:
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             for (x, y, w, h) in faces:
#                 face = gray[y:y + h, x:x + w]
#                 resized_face = cv2.resize(face, (128, 128))
#                 cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), resized_face)
#                 count += 1
#             if count >= num_images:
#                 break
#         cap.release()

#         return jsonify({"message": f"{count} images captured successfully!"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/train_model', methods=['POST'])
# def train_model():
#     try:
#         data = []
#         labels = []
#         for root, _, files in os.walk(DATASET_DIR):
#             for file in files:
#                 if file.endswith(".jpg"):
#                     image_path = os.path.join(root, file)
#                     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#                     image = cv2.resize(image, (128, 128))
#                     data.append(image)
#                     labels.append(os.path.basename(root))

#         data = np.array(data).reshape(-1, 128, 128, 1) / 255.0
#         label_encoder = LabelEncoder()
#         labels = label_encoder.fit_transform(labels)
#         np.save(LABEL_ENCODER_FILE, label_encoder.classes_)

#         X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#         model = Sequential([
#             Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
#             MaxPooling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu'),
#             MaxPooling2D((2, 2)),
#             Flatten(),
#             Dense(128, activation='relu'),
#             Dropout(0.5),
#             Dense(len(label_encoder.classes_), activation='softmax')
#         ])

#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
#         model.save(MODEL_FILE)

#         return jsonify({"message": "Model training complete!"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/mark_attendance', methods=['POST'])
# def mark_attendance():
#     try:
#         model = tf.keras.models.load_model(MODEL_FILE)
#         label_classes = np.load(LABEL_ENCODER_FILE, allow_pickle=True)

#         cap = cv2.VideoCapture(0)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         attendance = pd.read_csv(ATTENDANCE_FILE)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             for (x, y, w, h) in faces:
#                 face = gray[y:y + h, x:x + w]
#                 resized_face = cv2.resize(face, (128, 128)) / 255.0
#                 predictions = model.predict(np.expand_dims(resized_face, axis=(0, -1)))
#                 class_id = np.argmax(predictions)
#                 student_name = label_classes[class_id]

#                 timestamp = datetime.datetime.now()
#                 new_entry = {
#                     'Date': timestamp.date(),
#                     'Time': timestamp.time(),
#                     'ID': student_name,
#                     'Name': student_name
#                 }
#                 attendance = pd.concat([attendance, pd.DataFrame([new_entry])], ignore_index=True)
#                 attendance.to_csv(ATTENDANCE_FILE, index=False)
#             break
#         cap.release()
#         return jsonify({"message": "Attendance marked successfully!"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':  # Fixed syntax issue here
#     app.run(debug=True,port=5020)