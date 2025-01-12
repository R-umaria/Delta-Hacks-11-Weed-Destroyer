
import cv2
import numpy as np
import math
import keras
from PIL import Image
import sys

sys.stdout.reconfigure(encoding='utf-8')

sys.stderr.reconfigure(encoding='utf-8')

print(keras.__version__)

# Load the YOLOv8 model

uniformModel = keras.saving.load_model("multiselectRemoveHard1736.keras")

probability_model = keras.Sequential([uniformModel, 
                                         keras.layers.Softmax()])

uniformModel.summary()


# Open the video file
video_path = 0 
cap = cv2.VideoCapture(video_path)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        resized_image = cv2.resize(frame, (250, 250))


        cropped = resized_image.astype("uint8") # "float32")


        # cv2.imshow("YOLOv8 Tracking", cropped)
        

        print(cropped.shape)

        cropped = np.expand_dims(cropped, axis=0)

        print(cropped.shape)

        predictions = probability_model.predict(cropped)# , batch_size=1)

        uniform = np.argmax(predictions)

        prob_text = f"Weed{predictions[0][0]*100:.2f}|Crop{predictions[0][1]*100:.2f}"

        # Choose the text position (just above the top-left corner of the box)
        text_position = (100, 100)
        # print(text_position)

        # cv2.putText(frame, prob_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        print(predictions)


        print(predictions)
        print(uniform)

        if uniform != 1: 
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
            print("Crop: True")
        elif (uniform <= 0.5):
            print("Weeeeeed!")        
        else: 
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
            print("Ground")

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
