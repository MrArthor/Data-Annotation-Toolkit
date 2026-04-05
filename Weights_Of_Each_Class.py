import torch
from ultralytics import YOLO

class_counts = torch.tensor([31027, 24321, 15300], dtype=torch.float32)
total = class_counts.sum()
weights = total / (3 * class_counts)
print("Class Counts:", class_counts)
print("Total Annotations:", total)
print("Calculated Weights:", weights)
# Normalized weights: [0.83, 0.90, 1.47]