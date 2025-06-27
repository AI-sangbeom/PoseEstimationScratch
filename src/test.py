import torch 
import timm 
from models.custom_yolo import YOLOv11MBackbone

model = timm.create_model('resnet50', num_classes=0)


# 1. Get the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 2. Get the size of the model in bytes (approximated)
# Each parameter is typically a float32, which is 4 bytes.
param_size_bytes = total_params * 4
print(f"Approximate model size in bytes: {param_size_bytes}")
print(f"Approximate model size in kilobytes: {param_size_bytes / 1024}")
print(f"Approximate model size in megabytes: {param_size_bytes / (1024 * 1024)}")
