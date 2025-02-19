import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('best (3).pt')  # You can replace 'yolov8n.pt' with your custom model path if needed

# Read the image
image_path = 'IMG_4832.jpg'  # Replace with the path to your image

frame = cv2.imread(image_path)


# Perform detection
results = model(frame)[0]

# Draw bounding boxes on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > 0.5:  # You can adjust the threshold as needed
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert BGR image to RGB for displaying with matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.imshow(frame_rgb)
plt.axis('off')
plt.show()
