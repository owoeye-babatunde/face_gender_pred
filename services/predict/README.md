Technical Implementation Report: Gender and Race Classification API
Last Updated: December 30, 2024

1. API Architecture Overview

The API is built using FastAPI framework and implements two pre-trained ResNet models for gender and race classification. The system processes images asynchronously and returns probability distributions for both classifications.

2. Model Implementation

2.1 Gender Classification Model
- Architecture: ResNet18
- Output Classes: 2 (Male, Female)
- Model Loading:
```python
def load_gender_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load('model_checkpoint.pth', 
                          map_location=DEVICE, 
                          weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

2.2 Race Classification Model
- Architecture: ResNet50
- Output Classes: 6 (Asian, Black, Indian, Latino_Hispanic, Middle_Eastern, White)
- Model Loading:
```python
def load_race_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    checkpoint = torch.load('best_model.pth', 
                          map_location=DEVICE, 
                          weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

3. Image Preprocessing

3.1 Gender Model Preprocessing
```python
gender_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

3.2 Race Model Preprocessing
```python
race_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

4. Prediction Pipeline

4.1 Asynchronous Prediction Implementation
```python
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    gender_task = asyncio.create_task(predict_gender(image))
    race_task = asyncio.create_task(predict_race(image))
    
    gender_probs, race_probs = await asyncio.gather(
        gender_task, 
        race_task
    )
    
    return PredictionResponse(
        gender_probabilities=gender_probs,
        race_probabilities=race_probs
    )
```

4.2 Model Inference Functions
```python
async def predict_gender(image: Image.Image):
    img_tensor = gender_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = gender_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {GENDER_CLASSES[i]: float(prob) 
            for i, prob in enumerate(probabilities)}

async def predict_race(image: Image.Image):
    img_tensor = race_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = race_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {RACE_CLASSES[i]: float(prob) 
            for i, prob in enumerate(probabilities)}
```

5. Docker Implementation

5.1 API Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .
COPY model_checkpoint.pth .
COPY best_model.pth .

EXPOSE 8000
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
```

5.2 Docker Compose Configuration
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./model_checkpoint.pth:/app/model_checkpoint.pth
      - ./best_model.pth:/app/best_model.pth
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/docs"]
      interval: 30s
      timeout: 10s
      retries: 3
```

6. Dependencies

6.1 Core Requirements
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
torch==2.1.1
torchvision==0.16.1
Pillow==10.1.0
```

7. API Response Format
```json
{
    "gender_probabilities": {
        "Female": 0.92,
        "Male": 0.08
    },
    "race_probabilities": {
        "Asian": 0.15,
        "Black": 0.05,
        "Indian": 0.70,
        "Latino_Hispanic": 0.05,
        "Middle_Eastern": 0.03,
        "White": 0.02
    }
}
```

8. Technical Specifications

- Input Format: JPG, JPEG, PNG
- Image Processing: RGB, 224x224 pixels
- GPU Acceleration: Automatic if available
- Parallel Processing: Enabled via asyncio
- Docker Container: Isolated environment
- Health Monitoring: Integrated checks
