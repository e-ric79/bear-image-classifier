import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bear Classifier", page_icon="🐻", layout="centered")

BASE = Path(__file__).parent


# ── Load class names ──────────────────────────────────────────────────────────
@st.cache_resource
def load_classes():
    with open(BASE / "classes.json") as f:
        return json.load(f)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = torch.load(
        BASE / "bear_model_full.pth", map_location="cpu", weights_only=False
    )
    model.eval()
    return model


# ── Image transforms (must match fast.ai defaults) ───────────────────────────
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(img, model, classes):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], idx, probs


# ── Load everything ───────────────────────────────────────────────────────────
classes = load_classes()
model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐻 Bear Image Classifier")
st.markdown(
    "Upload a photo of a bear and the model will classify it as "
    "**Grizzly**, **Black**, or **Teddy**."
)
st.markdown("> Built with ResNet18 + fast.ai · Error rate: **0.048** · ")
st.divider()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Classifying..."):
        pred_class, pred_idx, probs = predict(img, model, classes)

    st.divider()

    EMOJIS = {"grizzly": "🐻", "black": "🐾", "teddy": "🧸"}
    emoji = EMOJIS.get(pred_class.lower(), "🐻")
    confidence = float(probs[pred_idx]) * 100

    st.markdown(f"### {emoji} Prediction: **{pred_class.capitalize()} Bear**")

    if confidence >= 85:
        st.success(f"Confidence: {confidence:.1f}%")
    elif confidence >= 60:
        st.warning(f"Confidence: {confidence:.1f}% — model is a little unsure")
    else:
        st.error(f"Confidence: {confidence:.1f}% — model is not confident")

    # ── Probability bar chart ─────────────────────────────────────────────────
    labels = [c.capitalize() for c in classes]
    values = [float(p) * 100 for p in probs]
    colors = ["#e07b39", "#4a4a4a", "#c9a96e"]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Confidence per class",
        yaxis=dict(title="Probability (%)", range=[0, 110]),
        xaxis=dict(title="Bear type"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Fun fact ──────────────────────────────────────────────────────────────
    facts = {
        "grizzly": "Grizzly and black bears are the model's hardest challenge — they share similar fur textures. Reflected in the confusion matrix.",
        "black": "Black bears were occasionally confused with grizzlies during training — similar colouring in certain lighting.",
        "teddy": "Teddy bears had 0 misclassifications during training. The model clearly learned the difference between real and toy bears!",
    }
    st.info(f"💡 {facts.get(pred_class.lower(), '')}")

else:
    st.markdown(
        "#### Try these bear types:\n- 🐻 Grizzly bear\n- 🐾 Black bear\n- 🧸 Teddy bear"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:grey; font-size:13px'>"
    "Built by Eric · CS Student (AI/ML) · Kenya 🇰🇪 · "
    "<a href='https://github.com/e-ric79/bear-image-classifier' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
