import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
# near other imports
from google.cloud.firestore_v1.base_query import FieldFilter, BaseCompositeFilter
from google.cloud.firestore_v1.types import StructuredQuery

# Initialize Firebase (keeps your existing secrets usage)
cred = credentials.Certificate({
    "type": st.secrets.firebase.type,
    "project_id": st.secrets.firebase.project_id,
    "private_key_id": st.secrets.firebase.private_key_id,
    "private_key": st.secrets.firebase.private_key,
    "client_email": st.secrets.firebase.client_email,
    "client_id": st.secrets.firebase.client_id,
    "auth_uri": st.secrets.firebase.auth_uri,
    "token_uri": st.secrets.firebase.token_uri,
    "auth_provider_x509_cert_url": st.secrets.firebase.auth_provider_x509_cert_url,
    "client_x509_cert_url": st.secrets.firebase.client_x509_cert_url
})

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_user_data(user_id, name, weight, height, age, gender, goal, daily_calories):
    """
    Save/update user under document id = user_id.
    Ensures fields: user_id, name, weight, height, age, gender, goal, daily_calories
    """
    if not user_id:
        raise ValueError("user_id required")
    doc_ref = db.collection("users").document(user_id)
    data = {
        "user_id": user_id,
        "name": name,
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender,
        "goal": goal,
        "daily_calories": daily_calories,
    }
    try:
        doc_ref.set(data, merge=True)
    except Exception as e:
        print(f"[save_user_data] Error: {e}")
        try:
            db.collection("users").add(data)
            print("[save_user_data] Fallback: created auto-id user doc")
        except Exception as e2:
            print(f"[save_user_data] Fallback failed: {e2}")

def get_user_data(user_id):
    """
    Return the user dict or None.
    1) try document(user_id)
    2) fallback: query where('user_id' == user_id) for legacy docs
    """
    if not user_id:
        return None
    users_ref = db.collection("users")

    # try direct document id first (keep as before)
    try:
        doc = users_ref.document(user_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        print(f"[get_user_data] doc fetch error: {e}")

    # fallback: use FieldFilter to avoid positional-arg warning
    try:
        q = users_ref.where(filter=FieldFilter("user_id", "==", user_id)).limit(1).stream()
        for d in q:
            return d.to_dict()
    except Exception as e:
        print(f"[get_user_data] query fallback error: {e}")

    print(f"[get_user_data] No user found for {user_id}")
    return None

def save_meal(user_id, food_name, calories, protein, fat, carbs, date):
    try:
        db.collection("meals").add({
            "user_id": user_id,
            "food_name": food_name,
            "calories": calories,
            "protein": protein,
            "fat": fat,
            "carbs": carbs,
            "date": date
        })
    except Exception as e:
        print(f"[save_meal] Error: {e}")

def get_meals_by_date(user_id, date):
    try:
        meals_ref = db.collection("meals")
        # build an AND composite filter to avoid positional-arg warnings
        filters = [
            FieldFilter("user_id", "==", user_id),
            FieldFilter("date", "==", date)
        ]
        comp = BaseCompositeFilter(StructuredQuery.CompositeFilter.Operator.AND, filters)
        query = meals_ref.where(filter=comp)
        return [doc.to_dict() for doc in query.stream()]
    except Exception as e:
        print(f"[get_meals_by_date] Error: {e}")
        return []
def save_undef_image(user_id, food_name, calories, protein, fat, carbs, date, thumb_path=None, predicted_conf=None, model_label=None):
    """
    Save an 'undefined/uncertain' image record into the 'undef_images' collection.
    Stores extra metadata: thumb path, predicted_confidence, model_label and created_at timestamp.
    """
    try:
        data = {
            "user_id": user_id,
            "food_name": food_name,
            "calories": calories,
            "protein": protein,
            "fat": fat,
            "carbs": carbs,
            "date": date,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        if thumb_path:
            data["thumb"] = thumb_path
        if predicted_conf is not None:
            # store confidence as float 0..1
            data["predicted_confidence"] = float(predicted_conf)
        if model_label:
            data["model_label"] = model_label
        db.collection("undef_images").add(data)
    except Exception as e:
        print(f"[save_undef_image] Error: {e}")
