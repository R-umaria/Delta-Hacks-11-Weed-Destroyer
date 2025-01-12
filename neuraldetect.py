import cv2
import numpy as np
from keras.saving import load_model
from keras.layers import Softmax
from keras import Sequential
import sys

# Ensure the terminal supports UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = load_model("multiselectRemoveHard1736.keras")
probability_model = Sequential([model, Softmax()])

model.summary()

# Open the video feed (0 for webcam)
video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    # Resize and preprocess the frame
    resized_image = cv2.resize(frame, (250, 250))
    cropped = np.expand_dims(resized_image.astype("uint8"), axis=0)

    # Make predictions
    predictions = probability_model.predict(cropped)
    class_index = np.argmax(predictions)

    # Interpret predictions
    prob_text = (
        f"Weed: {predictions[0][1]*100:.2f}%% | "
        f"Crop: {predictions[0][0]*100:.2f}%% | "
    )
    print(prob_text)

    if class_index == 0:
        print("Crop detected")
    elif class_index == 1:
        print("Weed detected")
    else:
        print("Ground detected")

# Release resources
cap.release()
cv2.destroyAllWindows()
