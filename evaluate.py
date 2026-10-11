import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

print("=" * 55)
print("VISIONCLASS AI - MODEL INFORMATION")
print("=" * 55)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load pretrained model
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model = model.to(device)
model.eval()

# Model information
categories = weights.meta["categories"]

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"\nModel: MobileNetV2")
print(f"Framework: PyTorch")
print(f"Device: {device}")

print(
    f"Number of classes: "
    f"{len(categories)}"
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print("\nPretrained Dataset: ImageNet-1K")

print(
    "Input Size: 224 x 224 RGB"
)

print("\n" + "=" * 55)
print("OFFICIAL PRETRAINED WEIGHT METRICS")
print("=" * 55)

# Metrics stored in torchvision weight metadata
metrics = weights.meta.get("_metrics", {})

imagenet_metrics = metrics.get(
    "ImageNet-1K",
    {}
)

top1 = imagenet_metrics.get(
    "acc@1",
    "N/A"
)

top5 = imagenet_metrics.get(
    "acc@5",
    "N/A"
)

print(
    f"\nTop-1 Accuracy: {top1}%"
)

print(
    f"Top-5 Accuracy: {top5}%"
)

print("\n" + "=" * 55)
print("MODEL READY")
print("=" * 55)