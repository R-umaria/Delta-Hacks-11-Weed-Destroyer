import RPi.GPIO as GPIO
import cv2
import numpy as np
from keras.saving import load_model
from keras.layers import Softmax
from keras import Sequential
import time

# ---------------------- GPIO Setup ----------------------
GPIO.setmode(GPIO.BCM)

# Pin configuration
motor_pins = {
    'left1': {'dir1': 5, 'dir2': 6, 'pwm': 13},  # Left motor 1
    'left2': {'dir1': 17, 'dir2': 27, 'pwm': 22},  # Left motor 2
    'right1': {'dir1': 11, 'dir2': 9, 'pwm': 10},  # Right motor 1
    'right2': {'dir1': 4, 'dir2': 3, 'pwm': 2},  # Right motor 2
}

# Initialize all motor pins
for motor in motor_pins.values():
    GPIO.setup(motor['dir1'], GPIO.OUT)
    GPIO.setup(motor['dir2'], GPIO.OUT)
    GPIO.setup(motor['pwm'], GPIO.OUT)
    motor['pwm_instance'] = GPIO.PWM(motor['pwm'], 1000)  # Set PWM frequency to 1kHz
    motor['pwm_instance'].start(0)  # Start with motor off

# ---------------------- Motor Control ----------------------
def move(direction, speed):
    """
    Controls the movement of the rover.
    :param direction: "forward" or "stop".
    :param speed: Speed percentage (0-100).
    """
    if direction == "forward":
        for motor in motor_pins.values():
            GPIO.output(motor['dir1'], GPIO.HIGH)
            GPIO.output(motor['dir2'], GPIO.LOW)
            motor['pwm_instance'].ChangeDutyCycle(speed)
    elif direction == "stop":
        for motor in motor_pins.values():
            GPIO.output(motor['dir1'], GPIO.LOW)
            GPIO.output(motor['dir2'], GPIO.LOW)
            motor['pwm_instance'].ChangeDutyCycle(0)

# ---------------------- Load Neural Network ----------------------
model = load_model("multiselectRemoveHard1736.keras")
probability_model = Sequential([model, Softmax()])

# ---------------------- Inference & Control Loop ----------------------
try:
    # Open the webcam feed
    cap = cv2.VideoCapture(0)

    # Start the rover in forward motion
    move("forward", 50)
    print("Rover moving forward...")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # Preprocess the frame
        resized_image = cv2.resize(frame, (250, 250))
        normalized_image = resized_image / 255.0  # Normalize to [0, 1]
        cropped = np.expand_dims(normalized_image.astype("float32"), axis=0)

        # Run predictions
        predictions = probability_model.predict(cropped)
        class_index = np.argmax(predictions)

        # Display predictions for debugging
        prob_text = (
            f"Weed: {predictions[0][0]*100:.2f}% | "
            f"Crop: {predictions[0][1]*100:.2f}% | "
            f"Ground: {predictions[0][2]*100:.2f}%"
        )
        print(prob_text)

        # Stop the rover if weed is detected
        if class_index == 0:  # Weed detected
            print("Weed detected! Stopping the rover.")
            move("stop")
            time.sleep(1)  # Pause for a second
        else:
            # Continue moving forward if no weed is detected
            print("No weed detected. Moving forward...")
            move("forward", 50)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    # Release resources
    cap.release()
    move("stop")
    GPIO.cleanup()
    print("Rover stopped, and GPIO cleaned up.")
