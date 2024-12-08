import cv2

import cv2
import threading
import queue
import time

# Create a queue to communicate between threads
frame_queue = queue.Queue()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Function to capture frames and perform processing (runs in a separate thread)
def capture_and_process():
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        # Perform any processing here (e.g., face recognition)
        processed_frame = cv2.putText(frame, "Processing Frame", (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Put the processed frame in the queue for the main thread to display
        if not frame_queue.full():
            frame_queue.put(processed_frame)
        time.sleep(0.03)  # Simulate processing delay

# Start the processing thread
thread = threading.Thread(target=capture_and_process, daemon=True)
thread.start()

# Main thread for displaying the frames
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        # Display the frame (must be in the main thread)
        cv2.imshow("Processed Video", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# # Load a simple image
# img = cv2.imread('Resources/background.png')

# if img is None:
#     print("Error: 'Resources/background.png' failed to load.")
# else:
#     print(f"Image shape: {img.shape}, dtype: {img.dtype}")
#     try:
#         cv2.imshow("Test Display", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except cv2.error as e:
#         print(f"OpenCV Error: {e}")
#     except Exception as e:
#         print(f"General Exception: {e}")
