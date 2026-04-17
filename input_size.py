from ultralytics import YOLO

# Load your model
model = YOLO("Architecture/x86/Yolo_Models/Helmet_Detection_Model/Models/Yolov8s/30032026_Yolov8s/results/helmet-detection-dataset/Project/train/weights/best.pt")

# Method 1: Get the training image size from metadata
train_imgsz = model.overrides.get('imgsz')
print(f"Training Input Size: {train_imgsz}")

# Method 2: Get the actual input shape from the network architecture
# YOLO models typically have a stride (e.g., 32)
stride = int(model.stride.max())
print(f"Model Stride: {stride}")

# Method 3: View full model configuration
# print(model.info())