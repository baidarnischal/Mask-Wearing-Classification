from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

app = FastAPI()

# =========================
# CORS (optional here since frontend is served from same origin)
# =========================
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load model
# =========================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "3_hypertuned_250_epochs" / "best_mask_cnn_hypertuned_250epochs.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
# MODEL = tf.keras.models.load_model("../models/3_hypertuned_250_epochs/best_mask_cnn_hypertuned_250epochs.keras")
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# =========================
# Serve frontend folder
# =========================
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

# Root route redirects to frontend
@app.get("/")
async def root():
    return RedirectResponse(url="/frontend/index.html")

# =========================
# Health check endpoint
# =========================
@app.get("/ping")
async def ping():
    return "hello"

# =========================
# Helper function
# =========================
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# =========================
# Run the app
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
