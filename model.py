import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── CLIP 모델 로드 ────────────────────────────────────
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# ── 이미지 로드 ───────────────────────────────────────
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def load_image_from_file(file):
    image = Image.open(file).convert("RGB")
    return image

# ── 이미지-텍스트 유사도 계산 ─────────────────────────
def compute_similarity(model, processor, image, texts):
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).numpy()[0]
    
    return probs

# ── 이미지 임베딩 ─────────────────────────────────────
def get_image_embedding(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.numpy()[0]

# ── 텍스트 임베딩 ─────────────────────────────────────
def get_text_embedding(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.numpy()[0]