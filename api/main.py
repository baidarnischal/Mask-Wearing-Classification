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

# ============================================================
# Initialize FastAPI app
# ============================================================
app = FastAPI(title="Mask Wearing Classification API")

# ============================================================
# CORS setup (allow all origins for simplicity)
# ============================================================
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# TensorFlow / Model setup
# ============================================================

# Optional: disable XLA JIT to avoid internal convolution errors
tf.config.optimizer.set_jit(False)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to trained model
MODEL_PATH = BASE_DIR / "models" / "3_hypertuned_250_epochs" / "best_mask_cnn_hypertuned_250epochs.keras"

# Determine if running in CI environment
CI_ENV = os.getenv("CI", "false").lower() == "true"

if CI_ENV:
    print("CI environment detected — using mock model")
    class MockModel:
        # Fake input shape: (batch_size, height, width, channels)
        input_shape = (1, 128, 128, 3)  
        
        def predict(self, x):
            import numpy as np
            # Return mock predictions
            # Must match the number of classes
            return np.array([[0.2, 0.5, 0.3]])
    MODEL = MockModel()
else:
    # Load model normally
    MODEL = tf.keras.models.load_model(MODEL_PATH)


# Your classes (must match training order)
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# ============================================================
# Helper: read and preprocess uploaded image
# ============================================================
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")

    # Auto-detect expected model input size (e.g., (128, 128, 3))
    target_size = MODEL.input_shape[1:3]

    # Resize to model input size
    image = image.resize(target_size)

    # Normalize to [0, 1] if that’s how you trained the model
    image = np.array(image) / 255.0

    return image

# ============================================================
# Serve frontend (optional)
# ============================================================
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
async def root():
    """Redirect to frontend index.html"""
    return RedirectResponse(url="/frontend/index.html")

# ============================================================
# Health check endpoint
# ============================================================
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API is alive"}

# ============================================================
# Prediction endpoint
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an image, run inference, and return the predicted class."""
    # Read uploaded file
    image = read_file_as_image(await file.read())

    # Expand batch dimension: (1, height, width, channels)
    img_batch = np.expand_dims(image, axis=0)

    # Run inference
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# ============================================================
# Run locally
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
