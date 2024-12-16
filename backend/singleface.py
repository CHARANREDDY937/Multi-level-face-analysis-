# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential, model_from_json
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from pymongo import MongoClient
# import gridfs

# # Parameters
# IMG_SIZE = 128  # Resize face images to 128x128
# EPOCHS = 20
# BATCH_SIZE = 16

# # MongoDB Configuration
# client = MongoClient("mongodb+srv://chary296:database@cluster.ru5lk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster")
# db = client["FaceRecognition_single"]
# fs = gridfs.GridFS(db)
# users_collection = db["users"]
# model_collection = db["model"]  # Collection for storing model

# # Initialize Haar Cascade for Face Detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def capture_faces(label_name, num_samples=100):
#     """
#     Captures face images using the webcam and saves them in MongoDB.
#     """
#     cap = cv2.VideoCapture(0)
#     count = 0

#     while count < num_samples:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             face = frame[y:y + h, x:x + w]
#             face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
#             # Convert image to binary data
#             _, img_encoded = cv2.imencode('.jpg', face)
#             img_binary = img_encoded.tobytes()

#             # Save to MongoDB
#             img_id = fs.put(img_binary, filename=f"{label_name}face{count}.jpg")
#             users_collection.update_one(
#                 {"label": label_name},
#                 {"$push": {"images": img_id}},
#                 upsert=True
#             )
#             count += 1

#             # Draw rectangle for feedback
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         cv2.imshow("Capturing Faces", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Collected {count} samples for label '{label_name}'.")

# def load_dataset():
#     """
#     Loads the dataset from MongoDB.
#     """
#     images, labels = [], []
#     class_names = []
    
#     # Fetch all users from the database
#     for user in users_collection.find():
#         label = user["label"]
#         class_names.append(label)
#         label_index = len(class_names) - 1
        
#         for img_id in user.get("images", []):
#             # Retrieve image from GridFS
#             img_binary = fs.get(img_id).read()
#             img_array = np.frombuffer(img_binary, np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#             images.append(img)
#             labels.append(label_index)

#     return np.array(images), np.array(labels), class_names

# # def save_model_to_mongodb(model):
# #     """
# #     Saves the model's architecture and weights to MongoDB.
# #     """
# #     # Convert model architecture to JSON
# #     model_json = model.to_json()

# #     # Save weights as binary
# #     weights = model.get_weights()
# #     weights_binary = [w.tobytes() for w in weights]

# #     # Save to MongoDB
# #     model_collection.replace_one(
# #         {"_id": "cnn_model"},
# #         {
# #             "_id": "cnn_model",
# #             "architecture": model_json,
# #             "weights": weights_binary
# #         },
# #         upsert=True
# #     )
# #     print("Model saved to MongoDB.")

# def save_model_to_mongodb(model):
#     """
#     Saves the model's architecture and weights to MongoDB.
#     """
#     architecture = model.to_json()
#     weights = [{"data": w.tobytes(), "shape": w.shape} for w in model.get_weights()]
#     weights_binary = [w["data"] for w in weights]
#     weights_shapes = [list(w["shape"]) for w in weights]

#     model_data = {
#         "_id": "cnn_model",
#         "architecture": architecture,
#         "weights": weights_binary,
#         "weights_shapes": weights_shapes,
#     }

#     # Upsert the model document in MongoDB
#     model_collection.update_one({"_id": "cnn_model"}, {"$set": model_data}, upsert=True)
#     print("Model saved to MongoDB.")


# # def load_model_from_mongodb():
# #     """
# #     Loads the model's architecture and weights from MongoDB.
# #     """
# #     model_data = model_collection.find_one({"_id": "cnn_model"})
# #     if not model_data:
# #         raise ValueError("Model not found in MongoDB.")

# #     # Load architecture
# #     model = model_from_json(model_data["architecture"])

# #     # Load weights
# #     weights_binary = model_data["weights"]
# #     weights = [np.frombuffer(w, dtype=np.float32).reshape(shape)
# #                for w, shape in zip(weights_binary, model.get_weights())]
# #     model.set_weights(weights)

# #     print("Model loaded from MongoDB.")
# #     return model

# def load_model_from_mongodb():
#     """
#     Loads the model's architecture and weights from MongoDB.
#     """
#     # Fetch the model document from MongoDB
#     model_data = model_collection.find_one({"_id": "cnn_model"})
#     if not model_data:
#         raise ValueError("Model not found in MongoDB.")

#     # Load architecture
#     model = model_from_json(model_data["architecture"])

#     # Load weights
#     weights_binary = model_data["weights"]
#     weights_shapes = model_data["weights_shapes"]  # Add this in save logic if missing

#     # Ensure weights are reshaped correctly
#     weights = [
#         np.frombuffer(w, dtype=np.float32).reshape(tuple(shape))
#         for w, shape in zip(weights_binary, weights_shapes)
#     ]
#     model.set_weights(weights)

#     print("Model loaded from MongoDB.")
#     return model


# def train_model():
#     """
#     Trains the CNN model using the dataset loaded from MongoDB.
#     """
#     images, labels, class_names = load_dataset()

#     if len(images) == 0:
#         raise ValueError("No data found! Capture faces first.")

#     # Preprocess Data
#     images = images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
#     labels = to_categorical(labels, num_classes=len(class_names))  # One-hot encode labels

#     # Split Data
#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Define the Model
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(len(class_names), activation='softmax')  # Output layer for face classes
#     ])

#     # Compile the Model
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     # Train the Model
#     model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

#     # Evaluate the Model
#     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#     print(f"Test Accuracy: {test_acc * 100:.2f}%")

#     # Save the Model to MongoDB
#     save_model_to_mongodb(model)

# def recognize_faces(threshold=0.8, confidence_gap=0.2):
#     """
#     Recognizes faces using the trained model and labels loaded from MongoDB.
#     """
#     model = load_model_from_mongodb()
#     class_names = sorted(users_collection.distinct("label"))
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             face = frame[y:y + h, x:x + w]
#             face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#             face = face.astype('float32') / 255.0
#             face = np.expand_dims(face, axis=0)

#             predictions = model.predict(face)
#             max_prob = np.max(predictions)
#             second_max_prob = np.partition(predictions.flatten(), -2)[-2] if len(predictions.flatten()) > 1 else 0
#             confidence_difference = max_prob - second_max_prob

#             if max_prob < threshold or confidence_difference < confidence_gap:
#                 predicted_class = "Unknown"
#             else:
#                 predicted_class = class_names[np.argmax(predictions)]

#             # Draw rectangle and label
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         cv2.imshow("Face Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Workflow
# if __name__ == "__main__":
#     print("1. Capture Faces and Train Model")
#     print("2. Recognize Faces")
#     choice = int(input("Enter your choice: "))

#     if choice == 1:
#         label = input("Enter label (person's name): ")
#         print("Step 1: Capturing faces...")
#         capture_faces(label_name=label)

#         print("Step 2: Training the model...")
#         class_names = train_model()

#         print("Process complete! You can now use the model for face recognition.")

#     elif choice == 2:
#         class_names = sorted(os.listdir(DATASET_PATH))
#         recognize_faces(class_names)
#     else:
#         print("Invalid choice!")

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pymongo import MongoClient
import gridfs


# Flask app initialization
app = Flask(__name__)
CORS(app)

# MongoDB configuration
client = MongoClient("mongodb+srv://chary296:database@cluster.ru5lk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster")  # Use your MongoDB URI here
db = client["FaceRecognition_singles"]
fs = gridfs.GridFS(db)
users_collection = db["users"]
model_collection = db["model"]

IMG_SIZE = 128  # Resize face images to 128x128
EPOCHS = 20
BATCH_SIZE = 16

# Initialize Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")









# Function to capture faces and save them to MongoDB
def capture_faces(label_name, num_samples=100):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
            # Convert image to binary data
            _, img_encoded = cv2.imencode('.jpg', face)
            img_binary = img_encoded.tobytes()

            # Save to MongoDB
            img_id = fs.put(img_binary, filename=f"{label_name}face{count}.jpg")
            users_collection.update_one(
                {"label": label_name},
                {"$push": {"images": img_id}},
                upsert=True
            )
            count += 1

            # Draw rectangle for feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} samples for label '{label_name}'.")


# Function to train the model using captured faces
def train_model():
    images, labels, class_names = [], [], []

    # Fetch all users from the database
    for user in users_collection.find():
        label = user["label"]
        class_names.append(label)
        label_index = len(class_names) - 1

        for img_id in user.get("images", []):
            # Retrieve image from GridFS
            img_binary = fs.get(img_id).read()
            img_array = np.frombuffer(img_binary, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(label_index)

    # Preprocess Data
    images = np.array(images).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(class_names))  # One-hot encode labels

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')  # Output layer for face classes
    ])

    # Compile the Model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the Model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # Evaluate the Model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save the Model to MongoDB
    architecture = model.to_json()
    weights = [{"data": w.tobytes(), "shape": w.shape} for w in model.get_weights()]
    weights_binary = [w["data"] for w in weights]
    weights_shapes = [list(w["shape"]) for w in weights]

    model_data = {
        "_id": "cnn_model",
        "architecture": architecture,
        "weights": weights_binary,
        "weights_shapes": weights_shapes,
    }

    model_collection.update_one({"_id": "cnn_model"}, {"$set": model_data}, upsert=True)
    print("Model saved to MongoDB.")


def load_model_from_mongodb():
    model_data = model_collection.find_one({"_id": "cnn_model"})
    if not model_data:
        raise ValueError("Model not found in MongoDB.")

    # Load model architecture
    model = model_from_json(model_data["architecture"])

    # Load weights
    weights_binary = model_data["weights"]
    weights_shapes = model_data["weights_shapes"]

    weights = [
        np.frombuffer(w, dtype=np.float32).reshape(tuple(shape))
        for w, shape in zip(weights_binary, weights_shapes)
    ]
    model.set_weights(weights)

    return model


# Function to recognize faces using webcam
def recognize_faces(threshold=0.8, confidence_gap=0.2):
    model = load_model_from_mongodb()
    class_names = sorted(users_collection.distinct("label"))
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)

            predictions = model.predict(face)
            max_prob = np.max(predictions)
            second_max_prob = np.partition(predictions.flatten(), -2)[-2] if len(predictions.flatten()) > 1 else 0
            confidence_difference = max_prob - second_max_prob

            if max_prob < threshold or confidence_difference < confidence_gap:
                predicted_class = "Unknown"
            else:
                predicted_class = class_names[np.argmax(predictions)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# API route to start face recognition
@app.route('/recognize_faces', methods=['GET'])
def recognize_faces_route():
    try:
        recognize_faces()
        return jsonify({"message": "Face recognition completed."}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500




# API route to start capturing faces
@app.route('/capture_faces', methods=['POST'])
def capture_faces_route():
    label_name = request.json.get('label_name')
    num_samples = request.json.get('num_samples', 100)
    
    try:
        capture_faces(label_name, num_samples)
        return jsonify({"message": "Faces captured successfully."}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


# API route to train the model
@app.route('/train_model', methods=['POST'])
def train_model_route():
    try:
        train_model()
        return jsonify({"message": "Model trained successfully."}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)