import torch
import clip
from PIL import Image
from torchvision import transforms

# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Define a function to preprocess the question and image
def preprocess_data(image_path, question):
    # Preprocess image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Preprocess question
    text_input = clip.tokenize([question]).to(device)
    
    return image_input, text_input

# Define a function to predict the answer
def predict_answer(image_path, question):
    image_input, text_input = preprocess_data(image_path, question)
    
    # Get features from the CLIP model
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Compute similarity between image and text features
    similarity = (image_features @ text_features.T).squeeze(0)
    return similarity
