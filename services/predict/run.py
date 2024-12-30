# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
import io
import asyncio
import numpy as np
from pydantic import BaseModel
from typing import Dict, List

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

# Initialize models
def load_gender_model():
    """Initialize and load the gender classification model (ResNet18)"""
    model = models.resnet18(weights=None)  # Initialize without pretrained weights
    # Modify the final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Load the saved state dict
    checkpoint = torch.load('model_checkpoint.pth', map_location=DEVICE, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def load_race_model():
    """Initialize and load the race classification model (ResNet50)"""
    model = models.resnet50(weights=None)  # Initialize without pretrained weights
    # Modify the final layer for 6-class classification
    model.fc = nn.Linear(model.fc.in_features, 6)
    # Load the saved state dict
    checkpoint = torch.load('best_model.pth', map_location=DEVICE, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

# Load and prepare models
gender_model = load_gender_model()
race_model = load_race_model()
gender_model.to(DEVICE)
race_model.to(DEVICE)
gender_model.eval()
race_model.eval()

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

class PredictionResponse(BaseModel):
    gender_probabilities: Dict[str, float]
    race_probabilities: Dict[str, float]

async def predict_gender(image: Image.Image) -> Dict[str, float]:
    """
    Predict gender probabilities from image
    """
    img_tensor = gender_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = gender_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {GENDER_CLASSES[i]: float(prob) for i, prob in enumerate(probabilities)}

async def predict_race(image: Image.Image) -> Dict[str, float]:
    """
    Predict race probabilities from image
    """
    img_tensor = race_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = race_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {RACE_CLASSES[i]: float(prob) for i, prob in enumerate(probabilities)}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict both gender and race from a single image
    """
    # Read and convert image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Run predictions in parallel
    gender_task = asyncio.create_task(predict_gender(image))
    race_task = asyncio.create_task(predict_race(image))
    
    # Wait for both predictions
    gender_probs, race_probs = await asyncio.gather(gender_task, race_task)
    
    return PredictionResponse(
        gender_probabilities=gender_probs,
        race_probabilities=race_probs
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("\nAccess the API at:")
    print("    http://localhost:8000")
    print("    or")
    print("    http://127.0.0.1:8000")