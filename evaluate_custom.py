from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# =========================
# SETTINGS
# =========================

DATASET_DIR = Path("evaluation")
OUTPUT_DIR = Path("screenshots")

TARGET_CLASSES = [
    "strawberry",
    "banana",
    "pineapple",
    "lemon",
    "orange",
]


# =========================
# LOAD MODEL
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(
    weights=weights
)

model = model.to(device)
model.eval()

preprocess = weights.transforms()

imagenet_classes = weights.meta["categories"]


# =========================
# CLASS MAPPING
# =========================

class_to_imagenet_index = {}

for class_name in TARGET_CLASSES:
    if class_name not in imagenet_classes:
        raise ValueError(
            f"{class_name} was not found in ImageNet classes."
        )

    class_to_imagenet_index[class_name] = (
        imagenet_classes.index(class_name)
    )


# =========================
# EVALUATION
# =========================

true_labels = []
predicted_labels = []

top1_correct = 0
top5_correct = 0
total_images = 0


for class_name in TARGET_CLASSES:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(
            f"WARNING: Missing folder: {class_dir}"
        )
        continue

    true_imagenet_index = class_to_imagenet_index[
        class_name
    ]

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
    ]:
        image_files.extend(
            class_dir.glob(extension)
        )

    print(
        f"\n{class_name}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            input_tensor = preprocess(
                image
            ).unsqueeze(0).to(device)

            with torch.inference_mode():

                output = model(
                    input_tensor
                )

                probabilities = torch.nn.functional.softmax(
                    output[0],
                    dim=0
                )

                top5_probabilities, top5_indices = (
                    torch.topk(
                        probabilities,
                        k=5
                    )
                )

            top1_prediction = (
                top5_indices[0].item()
            )

            top5_predictions = (
                top5_indices.tolist()
            )

            # Top-1 accuracy
            if top1_prediction == true_imagenet_index:
                top1_correct += 1

            # Top-5 accuracy
            if true_imagenet_index in top5_predictions:
                top5_correct += 1

            # Convert prediction into our 5-class
            # confusion-matrix label.
            if top1_prediction in [
                class_to_imagenet_index[c]
                for c in TARGET_CLASSES
            ]:
                predicted_class = next(
                    c
                    for c in TARGET_CLASSES
                    if class_to_imagenet_index[c]
                    == top1_prediction
                )
            else:
                predicted_class = "other"

            true_labels.append(class_name)
            predicted_labels.append(predicted_class)

            total_images += 1

        except Exception as error:
            print(
                f"Could not process "
                f"{image_path}: {error}"
            )


# =========================
# RESULTS
# =========================

if total_images == 0:
    print("\nNo images were found.")
    print(
        "Please check the evaluation folder."
    )
    raise SystemExit


top1_accuracy = (
    top1_correct / total_images
) * 100

top5_accuracy = (
    top5_correct / total_images
) * 100


print("\n")
print("=" * 60)
print("VISIONCLASS AI - CUSTOM EVALUATION")
print("=" * 60)

print(
    f"Total images: {total_images}"
)

print(
    f"Top-1 Accuracy: {top1_accuracy:.2f}%"
)

print(
    f"Top-5 Accuracy: {top5_accuracy:.2f}%"
)

print("=" * 60)


# =========================
# CONFUSION MATRIX
# =========================

matrix_labels = TARGET_CLASSES + ["other"]

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=matrix_labels
)

print("\nConfusion Matrix:")
print(cm)


# =========================
# SAVE CONFUSION MATRIX
# =========================

OUTPUT_DIR.mkdir(
    exist_ok=True
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=matrix_labels
)

display.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=45
)

ax.set_title(
    "MobileNetV2 Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

plt.tight_layout()

output_file = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:"
)

print(
    output_file
)

print("\nEvaluation completed successfully.")