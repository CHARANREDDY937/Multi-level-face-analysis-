import cv2
from flask import Flask, Response, jsonify, render_template
from lwcc import LWCC
from threading import Lock
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Initialize thread safety lock
lock = Lock()

# Initialize LWCC model globally
try:
    model = LWCC.load_model(model_name="DM-Count", model_weights="SHA")
    print("LWCC model loaded successfully.")
except Exception as e:
    print(f"Error loading LWCC model: {e}")
    exit()

# Global variables
cap = None
is_running = False


def generate_frames():
    """Video frame generator."""
    global cap, model

    while True:
        with lock:
            if not is_running or cap is None:
                break

            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

        # Save the frame to a temporary file for LWCC processing
        temp_file = "temp_frame.jpg"
        cv2.imwrite(temp_file, frame)

        # Process the frame with LWCC model
        try:
            crowd_count = LWCC.get_count(temp_file, model=model)
        except Exception as e:
            print(f"Error processing frame with LWCC: {e}")
            continue

        # Overlay the crowd count on the frame
        cv2.putText(frame, f"Estimated Crowd Count: {int(crowd_count)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Send frame to client
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('crowd.html')


@app.route('/video_feed')
def video_feed():
    """Stream video frames."""
    global is_running
    with lock:
        if not is_running:
            return Response("Video feed not started.", status=403)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_feed():
    """Start the video feed."""
    global cap, is_running
    with lock:
        if not is_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Cannot access webcam"}), 500
            is_running = True
    return jsonify({"status": "started"}), 200


@app.route('/stop', methods=['POST'])
def stop_feed():
    """Stop the video feed."""
    global cap, is_running
    with lock:
        is_running = False
        if cap:
            cap.release()
            cap = None
    return jsonify({"status": "stopped"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)