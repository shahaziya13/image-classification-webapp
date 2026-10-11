import io

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load Pretrained MobileNetV2
# --------------------------------------------------

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(
    weights=weights
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocess = weights.transforms()


# --------------------------------------------------
# ImageNet Class Names
# --------------------------------------------------

class_names = weights.meta[
    "categories"
]


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_image(
    image_bytes: bytes,
    top_k: int = 5
):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:

        raise ValueError(
            "The uploaded file is not a valid image."
        ) from exc


    # Preprocess image
    input_tensor = preprocess(
        image
    ).unsqueeze(0).to(device)


    # Prediction
    with torch.inference_mode():

        output = model(
            input_tensor
        )

        probabilities = (
            torch.nn.functional.softmax(
                output[0],
                dim=0
            )
        )


        top_probabilities, top_indices = (
            torch.topk(
                probabilities,
                k=top_k
            )
        )


    predictions = []


    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        predictions.append(
            {
                "class":
                    class_names[
                        index.item()
                    ],

                "confidence":
                    round(
                        probability.item()
                        * 100,
                        2
                    ),
            }
        )


    return predictions