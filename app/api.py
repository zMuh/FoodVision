from fastapi import FastAPI, UploadFile, File
from model.model import predict as model_predict
from app.utils import get_nutrition
import tempfile
import shutil

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Run prediction
    results = model_predict(
        weights="./models/best.pt",
        source=tmp_path,
        imgsz=640,
        conf=0.25,
        save=False
    )

    # Extract **food name** and confidence from YOLO classification
    food_name = "Unknown"
    confidence = 0.0
    for r in results:
        if hasattr(r, "probs"):
            top_idx = r.probs.top1
            food_name = r.names[top_idx]
            confidence = float(r.probs.top1conf)
            break

    # Get nutrition info using the class name directly
    food_name_clean = food_name.replace("_", " ")
    nutrition_info = get_nutrition(food_name_clean)

    # Placeholder for Firebase storage
    # firebase_store_meal({'food': food_name, 'confidence': confidence, 'nutrition': nutrition_info})
    # Implementation left for another developer

    # Delete temp file
    try:
        shutil.rmtree(tmp_path)
    except Exception:
        pass

    return {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition_info,
        "message": "Meal info can be stored in Firebase (placeholder)"
    }
    ''' Example usage
    ➜  FoodVision git:(main) ✗ curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/Users/muhannad/code/zMuh/FoodVision/raw_data/food-101/food101_yolo/val/spring_rolls/88158.jpg "

{"food":"spring_rolls","confidence":0.9886192083358765,"nutrition":{"calories":135,"protein":3.71,"fat":1.11,"carbs":27.1},"message":"Meal info can be stored in Firebase (placeholder)"}%'''
