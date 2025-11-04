import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from numpy.linalg import norm

st.title("Retail Product Matcher Demo")

# Load catalog
catalog_path = Path("catalog.csv")
catalog = pd.read_csv(catalog_path)
img_dir = Path("images")

# Load model
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model.features(img_t)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        return pooled.view(-1).numpy()

def cosine_similarity(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))

# Compute embeddings if not in catalog
if "embedding" not in catalog.columns:
    catalog["embedding"] = catalog["image"].apply(lambda x: get_features(img_dir / x))

uploaded_file = st.file_uploader("Upload a product image", type=["jpg","jpeg","png","jfif","webp","avif"])
if uploaded_file:
    query_embedding = get_features(uploaded_file)
    catalog["similarity"] = catalog["embedding"].apply(lambda e: cosine_similarity(e, query_embedding))
    match = catalog.sort_values("similarity", ascending=False).iloc[0]

    st.image(uploaded_file, caption="Query Image", use_column_width=True)
    st.success(f"Closest Match: {match['name']} (${match['current_price']})")
    st.info(f"Category: {match['category']}, Similarity Score: {match['similarity']:.3f}")
    st.image(img_dir / match['image'], caption="Matched Product", use_column_width=True)
