from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import get_model
import io
import argparse
import os

app = FastAPI()

# Mount a static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Start car angle classifier API")
parser.add_argument("--model", type=str, default="resnet50", 
                    choices=['resnet50', 'resnet18', 'efficientnet_b0', 'efficientnet_v2_s', 'vit_b_16', 'regnet_y_32gf'], 
                    help="Model architecture to use")
args = parser.parse_args()

# Load the trained model
model = get_model(num_classes=8, model_name=args.model)
model_path = os.path.join('models', args.model, f'best_model_{args.model}.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

idx_to_class = {0: '0', 1: '130', 2: '180', 3: '230', 4: '270', 5: '320', 6: '40', 7: '90'}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file contents
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)
    predicted_label = idx_to_class[predicted_idx.item()]

    return {"class_name": predicted_label, "confidence": torch.softmax(output, dim=1)[0][predicted_idx].item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
