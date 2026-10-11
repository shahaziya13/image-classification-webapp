import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

# Reduced dataset for CPU-friendly training
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# CIFAR-10 Classes
# --------------------------------------------------

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# --------------------------------------------------
# Image Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Load CIFAR-10 Dataset
# --------------------------------------------------

print("\nLoading CIFAR-10 dataset...")


full_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)


full_test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)


# --------------------------------------------------
# Create Smaller Dataset for Faster CPU Training
# --------------------------------------------------

train_dataset = Subset(
    full_train_dataset,
    range(TRAIN_SAMPLES)
)

test_dataset = Subset(
    full_test_dataset,
    range(TEST_SAMPLES)
)


print(
    f"\nTraining images: {len(train_dataset)}"
)

print(
    f"Test images: {len(test_dataset)}"
)


# --------------------------------------------------
# Data Loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# Load Pretrained MobileNetV2
# --------------------------------------------------

print("\nLoading pretrained MobileNetV2...")


weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(
    weights=weights
)


# --------------------------------------------------
# Freeze Feature Extractor
# --------------------------------------------------

for parameter in model.features.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# Replace Final Classification Layer
# --------------------------------------------------

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(CLASS_NAMES)
)


model = model.to(DEVICE)


# --------------------------------------------------
# Loss Function and Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Create Models Folder
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_accuracy = 0.0


print("\nStarting training...")


for epoch in range(EPOCHS):

    print(
        f"\n{'=' * 50}"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"{'=' * 50}"
    )


    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for batch_index, (images, labels) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update classifier weights
        optimizer.step()


        running_loss += loss.item()


        # Get predicted class
        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


        # --------------------------------------------------
        # Display Progress Every 20 Batches
        # --------------------------------------------------

        if (
            (batch_index + 1) % 20 == 0
            or
            (batch_index + 1) == len(train_loader)
        ):

            current_accuracy = (
                100 * correct / total
            )

            average_loss = (
                running_loss /
                (batch_index + 1)
            )


            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
                f" | Loss: "
                f"{average_loss:.4f}"
                f" | Accuracy: "
                f"{current_accuracy:.2f}%"
            )


    # --------------------------------------------------
    # Training Accuracy
    # --------------------------------------------------

    training_accuracy = (
        100 * correct / total
    )


    print(
        f"\nTraining Accuracy: "
        f"{training_accuracy:.2f}%"
    )


    # --------------------------------------------------
    # Evaluate on Test Dataset
    # --------------------------------------------------

    print(
        "Evaluating on test images..."
    )


    model.eval()


    test_correct = 0

    test_total = 0


    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            test_total += labels.size(0)


            test_correct += (
                predicted == labels
            ).sum().item()


    # --------------------------------------------------
    # Test Accuracy
    # --------------------------------------------------

    test_accuracy = (
        100 *
        test_correct /
        test_total
    )


    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )


    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy


        torch.save(
            model.state_dict(),
            "models/mobilenetv2_cifar10.pth",
        )


        print(
            "✓ New best model saved!"
        )


# --------------------------------------------------
# Training Complete
# --------------------------------------------------

print(
    f"\n{'=' * 50}"
)

print(
    "TRAINING COMPLETE"
)

print(
    f"{'=' * 50}"
)


print(
    f"Best Test Accuracy: "
    f"{best_accuracy:.2f}%"
)


print(
    "Model saved at: "
    "models/mobilenetv2_cifar10.pth"
)


print(
    "\nCIFAR-10 Classes:"
)


for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index}: {class_name}"
    )