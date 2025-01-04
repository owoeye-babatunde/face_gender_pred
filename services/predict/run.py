import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
import io
import os
import uvicorn
from typing import Dict
from s3_utils import S3Handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Gender and Race Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
GENDER_CLASSES = ['Female', 'Male']
RACE_CLASSES = ['Asian', 'Black', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'White']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Model configurations
gender_model = None
race_model = None
s3_handler = None

# Image transforms
gender_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

race_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gender_model_loaded": gender_model is not None,
        "race_model_loaded": race_model is not None,
        "device": str(DEVICE)
    }

async def load_gender_model():
    """Load gender model from S3 if not already loaded"""
    global gender_model
    if gender_model is None:
        try:
            logger.info("Loading gender model from S3...")
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            model_bytes = s3_handler.download_model('model_checkpoint.pth')
            logger.info(f"Downloaded gender model from S3, size: {len(model_bytes.getvalue())} bytes")
            
            checkpoint = torch.load(model_bytes, map_location=DEVICE)
            logger.info("Loaded checkpoint into memory")

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded gender model using model_state_dict")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded gender model using state_dict")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded gender model using direct state dict")
            
            model.eval()
            gender_model = model.to(DEVICE)
            logger.info(f"Gender model successfully loaded and moved to {DEVICE}")
            return True
        except Exception as e:
            logger.error(f"Error loading gender model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load gender model: {str(e)}")
    return False

async def load_race_model():
    """Load race model from S3 if not already loaded"""
    global race_model
    if race_model is None:
        try:
            logger.info("Loading race model from S3...")
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 6)
            
            model_bytes = s3_handler.download_model('best_model.pth')
            logger.info(f"Downloaded race model from S3, size: {len(model_bytes.getvalue())} bytes")
            
            checkpoint = torch.load(model_bytes, map_location=DEVICE)
            logger.info("Loaded checkpoint into memory")

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded race model using model_state_dict")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded race model using state_dict")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded race model using direct state dict")
            
            model.eval()
            race_model = model.to(DEVICE)
            logger.info(f"Race model successfully loaded and moved to {DEVICE}")
            return True
        except Exception as e:
            logger.error(f"Error loading race model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load race model: {str(e)}")
    return False

async def predict_gender(image: Image.Image) -> Dict[str, float]:
    """Predict gender probabilities from image"""
    if gender_model is None:
        await load_gender_model()
    img_tensor = gender_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = gender_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {GENDER_CLASSES[i]: float(prob) for i, prob in enumerate(probabilities)}

async def predict_race(image: Image.Image) -> Dict[str, float]:
    """Predict race probabilities from image"""
    if race_model is None:
        await load_race_model()
    img_tensor = race_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = race_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {RACE_CLASSES[i]: float(prob) for i, prob in enumerate(probabilities)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict both gender and race from a single image"""
    try:
        logger.info(f"Received prediction request for file: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        gender_probs = await predict_gender(image)
        race_probs = await predict_race(image)
        
        logger.info("Prediction completed successfully")
        return {
            "gender_probabilities": gender_probs,
            "race_probabilities": race_probs
        }
    except Exception as e:
        logger.error(f"Error occurred during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize S3 handler and load models on startup"""
    global s3_handler
    logger.info("Starting application...")
    try:
        s3_handler = S3Handler()
        await load_gender_model()
        await load_race_model()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    if s3_handler:
        s3_handler.cleanup()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("run:app", host="0.0.0.0", port=port, reload=False)