import cv2
import os
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt

original_dataset = "data/kaggle_dataset"
yolo_dataset = "data/yolo_dataset"

print("Creating directories in:", yolo_dataset)

subdirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
for subdir in subdirs:
    os.makedirs(os.path.join(yolo_dataset, subdir), exist_ok=True)

# Get class names and assign numerical labels
class_names = sorted(os.listdir(os.path.join(original_dataset, "train")))
class_dict = {name: i for i, name in enumerate(class_names)}

def convert_and_save_images_and_labels(split):
    """Convert dataset into YOLO format."""
    original_path = os.path.join(original_dataset, split)
    image_output_path = os.path.join(yolo_dataset, "images", split)
    label_output_path = os.path.join(yolo_dataset, "labels", split)

    for class_name in class_names:
        class_path = os.path.join(original_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist. Skipping.")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not os.path.isfile(img_path):
                continue
            
            # Read image and get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read {img_path}")
                continue
            
            h, w, _ = img.shape

            # Save image
            shutil.copy(img_path, os.path.join(image_output_path, img_name))

            # Create YOLO label (assuming object occupies full image)
            x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_output_path, label_name)

            with open(label_path, "w") as f:
                f.write(f"{class_dict[class_name]} {x_center} {y_center} {width} {height}\n")

# Check for converted Dataset
train_images_path = os.path.join(yolo_dataset, "images", "train")
if not os.path.exists(train_images_path) or len(os.listdir(train_images_path)) == 0:
    print("Converting dataset to YOLO format...")
    convert_and_save_images_and_labels("train")
    convert_and_save_images_and_labels("val")
    convert_and_save_images_and_labels("test")
else:
    print(f"YOLO dataset already exists in: {train_images_path}. Skipping conversion.")
    print(os.listdir(yolo_dataset))

# Generate data.yaml file
num_classes = len(class_names)
yaml_content = f"""path: {os.path.abspath(yolo_dataset)}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {class_names}
"""

with open(os.path.join(yolo_dataset, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Dataset is ready for YOLOv8 training.")

# Train Model
model_path = "joemodel.pt"
if not os.path.exists(model_path):
    print(f"{model_path} not found. Training model...")
    model = YOLO('joemodel.pt')
    model.train(data=os.path.join(yolo_dataset, "data.yaml"), epochs=50, batch=16, imgsz=640)
    model.save(model_path)
else:
    print(f"Model {model_path} already exists. Skipping training.")
    model = YOLO(model_path)

# Load Test Image
image_path = "test_images/orange_test.jpg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"Error: Unable to load image {image_path}")

# Inference
results = model(frame)[0]

print("Detected objects details:")
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    label = results.names[int(class_id)]
    print(f"Class: {label} | Confidence: {score:.2f} | Box: ({x1}, {y1}), ({x2}, {y2})")

# Check how many different classes are detected
detected_classes = set()
for result in results.boxes.data.tolist():
    class_id = int(result[5])  # class_id is typically at index 5 in YOLOv8
    detected_classes.add(class_id)

print("Classes detected:", [results.names[class_id] for class_id in detected_classes])

# Bounding boxes
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    print(result)
    if score > 0.5:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{results.names[int(class_id)].upper()} {score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Plot
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame_rgb)
plt.axis('off')
plt.show()
