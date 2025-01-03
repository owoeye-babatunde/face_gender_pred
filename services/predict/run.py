import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
import io
from typing import Dict
from s3_utils import S3Handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

# Model configurations
gender_model = None
race_model = None
s3_handler = S3Handler()

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

async def load_gender_model():
    """Load gender model from S3 if not already loaded"""
    global gender_model
    if gender_model is None:
        try:
            logger.info("Loading gender model from S3...")
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            # Download model as bytes
            model_bytes = s3_handler.download_model('model_checkpoint.pth')
            logger.info("Downloaded gender model from S3")
            
            # Load the model from bytes
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
            
            # Download model as bytes
            model_bytes = s3_handler.download_model('best_model.pth')
            logger.info("Downloaded race model from S3")
            
            # Load the model from bytes
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
    await load_gender_model()
    img_tensor = gender_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = gender_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {GENDER_CLASSES[i]: float(prob) for i, prob in enumerate(probabilities)}

async def predict_race(image: Image.Image) -> Dict[str, float]:
    """Predict race probabilities from image"""
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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        gender_probs = await predict_gender(image)
        race_probs = await predict_race(image)
        
        return {
            "gender_probabilities": gender_probs,
            "race_probabilities": race_probs
        }
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    await load_gender_model()
    await load_race_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temporary files on shutdown"""
    s3_handler.cleanup()