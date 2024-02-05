from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from gpiozero import Motor
import cv2
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

flmotor = Motor(forward=10, backward=12)
frmotor = Motor(forward=18, backward=17)
blmotor = Motor(forward=16, backward=13)
brmotor = Motor(forward=9, backward=11)

def left():
    flmotor.backward()
    frmotor.forward()
    blmotor.backward()
    brmotor.forward()

def right():
    flmotor.forward()
    frmotor.backward()
    blmotor.forward()
    brmotor.backward()

def forward():
    flmotor.forward()
    frmotor.forward()
    blmotor.forward()
    brmotor.forward()

def reverse():
    flmotor.backward()
    frmotor.backward()
    blmotor.backward()
    brmotor.backward()

def stop():
    flmotor.stop()
    frmotor.stop()
    blmotor.stop()
    brmotor.stop()

# Function to perform leaf health detection on live video stream
def detect_leaf_health(video_source=0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define a broader range for green color
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])

        # Define a range for yellow color
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create masks to extract green and yellow regions
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours in the green and yellow masks
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through green contours and draw bounding boxes
        for contour in contours_green:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out small and large regions
            if 1000 < w * h < 50000:
                # Calculate the green area within the bounding box
                roi_mask = mask_green[y:y+h, x:x+w]
                green_area = cv2.countNonZero(roi_mask)

                # Calculate the percentage of green area in the bounding box
                percentage_green = (green_area / (w * h)) * 100

                # Determine health status based on adjusted criteria
                health_status = "Healthy" if percentage_green >= 10 else "Unhealthy"

                # Draw bounding box, display health status, and percentage
                color = (0, 255, 0) if health_status == "Healthy" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"Leaf Health: {health_status} (Green: {percentage_green:.2f}%)"
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Iterate through yellow contours and draw bounding boxes
        for contour in contours_yellow:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out small and large regions
            if 1000 < w * h < 50000:
                # Calculate the yellow area within the bounding box
                roi_mask = mask_yellow[y:y+h, x:x+w]
                yellow_area = cv2.countNonZero(roi_mask)

                # Calculate the percentage of yellow area in the bounding box
                percentage_yellow = (yellow_area / (w * h)) * 100

                # Determine health status based on adjusted criteria
                health_status_yellow = "Unhealthy" if percentage_yellow >= 5 else "Healthy"

                # Draw bounding box, display health status, and percentage
                color = (0, 0, 255) if health_status_yellow == "Unhealthy" else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"Leaf Health: {health_status_yellow} (Yellow: {percentage_yellow:.2f}%)"
                cv2.putText(frame, text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Send the processed frame to the clients
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer)

        socketio.emit('frame', {'image': frame_encoded.decode('utf-8')})

    # Release the video capture object
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('motor_command')
def handle_motor_command(data):
    command = data['command']
    if command == 'forward':
        forward()
    elif command == 'reverse':
        reverse()
    elif command == 'left':
        left()
    elif command == 'right':
        right()
    elif command == 'stop':
        stop()

if __name__ == '__main__':
    socketio.start_background_task(target=detect_leaf_health)
    socketio.run(app, debug=True, use_reloader=False)

