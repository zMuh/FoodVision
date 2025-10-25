import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("app/firebase_key.json")
firebase_admin.initialize_app(cred)

# Connect to Firestore
db = firestore.client()

def save_user_data(user_id, name, weight, height, goal, daily_calories):
    """Save user profile to Firestore"""
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set({
        "name": name,
        "weight": weight,
        "height": height,
        "goal": goal,
        "daily_calories": daily_calories
    })

def save_meal(user_id, food_name, calories, protein, fat, carbs, date):
    """Save meal record to Firestore"""
    db.collection("meals").add({
        "user_id": user_id,
        "food_name": food_name,
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "carbs": carbs,
        "date": date
    })

def get_meals_by_date(user_id, date):
    """Retrieve meals for a user by date"""
    meals_ref = db.collection("meals")
    query = meals_ref.where("user_id", "==", user_id).where("date", "==", date)
    return [doc.to_dict() for doc in query.stream()]
