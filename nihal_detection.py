import cv2
import numpy as np

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

        # Display the result
        cv2.imshow('Leaf Health Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Call the function with the appropriate video source
# For Raspberry Pi Camera Module, use video_source=0
# For USB camera, use the appropriate video source (e.g., video_source=1)
detect_leaf_health(video_source=0)
