
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("models/forest_fire_cnn.keras")

# Open your computer's webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    print("Check camera permissions and close other apps using the camera.")
    raise SystemExit

print("Live camera started. Press Q to stop.")
predictions = []
while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Convert camera image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image to match the CNN input size
    resized = cv2.resize(rgb_frame, (128, 128))

    # Prepare image for prediction
    image_array = np.expand_dims(resized, axis=0).astype(np.float32)

    # Predict using the trained CNN
    prediction = float(model.predict(image_array, verbose=0)[0][0])
    predictions.append(prediction)
    # Keep only the latest 10 predictions
    if len(predictions) > 30:
        predictions.pop(0)
    # Average the latest predictions
    avg_prediction = sum(predictions) / len(predictions)

    # Class order: fire = 0, no fire = 1
    if avg_prediction < 0.5:
        label = "FIRE DETECTED"
        confidence = (1 - avg_prediction) * 100
        color = (0, 0, 255)
    else:
        label = "NO FIRE DETECTED"
        confidence = avg_prediction * 100
        color = (0, 255, 0)

    # Display prediction on the camera frame
    display_text = f"{label} | Confidence: {confidence:.1f}%"

    cv2.putText(
        frame,
        display_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )
    
    cv2.imshow("CNN Forest Fire Detection - Press Q to Exit", frame)

    # Press Q to stop the camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
