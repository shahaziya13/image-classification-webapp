from pathlib import Path

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_image


app = FastAPI(
    title="Image Classification API",
    description="Image classification API using pretrained MobileNetV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(
    directory="templates"
)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE = 5 * 1024 * 1024


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": "MobileNetV2",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    # Validate image type using MIME type OR extension
    if (
        file.content_type not in ALLOWED_TYPES
        and extension not in ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPG, JPEG, PNG, or WEBP.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum file size is 5 MB.",
        )

    try:
        predictions = predict_image(
            image_bytes,
            top_k=5
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image.",
        )

    return {
        "filename": filename,
        "model": "MobileNetV2",
        "predictions": predictions,
    }