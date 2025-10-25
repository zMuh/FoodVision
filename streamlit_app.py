import streamlit as st
import tempfile
import shutil
from ultralytics import YOLO
from model.model import predict as model_predict  # if you have a wrapper, else use YOLO directly
from app.utils import get_nutrition  # your nutrition lookup function

# --- Load YOLO model ---
MODEL_PATH = "./models/best.pt"
model = YOLO(MODEL_PATH)

st.title("🍽️ FoodVision Classifier")
st.write("Upload a food image to predict its class and nutrition info.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = tmp.name

    # Run prediction (same as FastAPI logic)
    results = model.predict(source=tmp_path, imgsz=640, conf=0.25, save=False)

    # Extract top class name + confidence
    food_name = "Unknown"
    confidence = 0.0
    for r in results:
        if hasattr(r, "probs"):
            top_idx = r.probs.top1
            food_name = r.names[top_idx]
            confidence = float(r.probs.top1conf)
            break

    # Get nutrition info
    food_name_clean = food_name.replace("_", " ")
    nutrition_info = get_nutrition(food_name_clean)

    # Display results
    st.image(uploaded_file, caption=f"Predicted: {food_name_clean} ({confidence:.2f})", use_container_width=True)
    st.markdown("### 🧠 Prediction Details")
    st.json({
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition_info,
        "message": "Meal info can be stored in Firebase (placeholder)"
    })

    # Clean up
    try:
        shutil.rmtree(tmp_path)
    except Exception:
        pass
