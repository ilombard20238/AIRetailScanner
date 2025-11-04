import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from numpy.linalg import norm

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Retail AI Scanner", layout="wide")
st.title("🛒 Retail Product Scanner Demo")

# ---------------------------
# Paths (relative to repo)
# ---------------------------
catalog_path = Path("catalog.csv")
img_dir = Path("images")

# ---------------------------
# Load catalog
# ---------------------------
catalog = pd.read_csv(catalog_path)

# ---------------------------
# Load pretrained AlexNet (cached)
# ---------------------------
@st.cache_resource
def load_model():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.eval()
    return model

model = load_model()

# ---------------------------
# Image transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------------
# Feature extraction
# ---------------------------
def get_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model.features(img_t)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        return pooled.view(-1).numpy()

# ---------------------------
# Cosine similarity
# ---------------------------
def cosine_similarity(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))

# ---------------------------
# Compute embeddings for catalog (cached)
# ---------------------------
@st.cache_data
def compute_embeddings():
    embeddings = []
    for img_name in catalog["image"]:
        emb = get_features(img_dir / img_name)
        embeddings.append(emb)
    return embeddings

if "embedding" not in catalog.columns:
    catalog["embedding"] = compute_embeddings()

# ---------------------------
# Upload query image
# ---------------------------
uploaded_file = st.file_uploader("Upload a product image to scan:", 
                                 type=["jpg","jpeg","png","jfif","webp","avif"])

if uploaded_file:
    query_embedding = get_features(uploaded_file)
    catalog["similarity"] = catalog["embedding"].apply(lambda e: cosine_similarity(e, query_embedding))
    match = catalog.sort_values("similarity", ascending=False).iloc[0]

    st.subheader("Query Image")
    st.image(uploaded_file, use_column_width=True)

    st.subheader("Closest Match")
    st.image(img_dir / match["image"], use_column_width=True)
    st.markdown(f"**Product:** {match['name']}")
    st.markdown(f"**Category:** {match['category']}")
    st.markdown(f"**Estimated Price:** ${match['current_price']}")
    st.markdown(f"**Similarity Score:** {match['similarity']:.3f}")
