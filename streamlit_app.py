
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import base64

# ------------------ Optional / external imports ------------------
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Firebase helpers detected at runtime if available
USE_FIREBASE = False
try:
    from app.firebase_utils import (
        save_user_data,
        save_meal,
        get_meals_by_date,
        get_user_data,
        save_undef_image,
    )
    USE_FIREBASE = True
except Exception:
    # Not fatal; fallback to in-memory store
    USE_FIREBASE = False

# App utils if provided
HAS_UTILS = False
try:
    from app.utils import get_nutrition, calculate_calories
    HAS_UTILS = True
except Exception:
    HAS_UTILS = False


# ------------------ Constants & Theme ------------------
APP_NAME = "FoodVision"
THUMBS_DIR = Path("thumbs")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best.pt"

# Color palette (used in CSS & charts)
WHITE = "#ffffff"
PRIMARY = "#24a691"
PRIMARY_DARK = "#5ab49f"
ACCENT = "#75c4a9"
NAV_BG = "#19212c"
BG_TOP = "#1c2d40"
BG_BOTTOM = "#28485d"
TEXT = "#f7f9fb"
CARD_BG = "rgba(27,45,66,0.72)"

# Ensure thumbs and models dirs exist
THUMBS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ------------------ Unified CSS ------------------
GLOBAL_CSS = f"""
<style>

:root {{
  --primary: {PRIMARY};
  --primary-dark: {PRIMARY_DARK};
  --accent: {ACCENT};
  --bg-top: {BG_TOP};
  --bg-bottom: {BG_BOTTOM};
  --nav-bg: {NAV_BG};
  --card-bg: {CARD_BG};
  --text: {TEXT};
  --white: {WHITE};
}}

/* App main background (dark blue gradient) */
html, body, .reportview-container .main, .block-container {{
  background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%) !important;
  color: var(--text) !important;
}}
.stApp {{
  min-height: 100vh;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%) !important;
  color: var(--text) !important;
}}

.stSidebar .sidebar-content {{
  background: linear-gradient(180deg, #8CD3BF 0%, #43B19D 100%) !important;
  color: var(--text) !important;
}}

.card {{
  background: var(--card-bg);
  border-radius: 12px;
  padding: 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}}

input[type="text"], input[type="password"], input[type="number"], textarea {{
  height: 36px !important;
  border-radius: 8px !important;
}}

button[role="button"] {{
  background: linear-gradient(90deg,var(--primary),var(--primary-dark)) !important;
  color: white !important;
  border-radius: 8px !important;
  padding: 6px 12px !important;
  font-weight: 600;
}}

footer {{
  color: #94a3b8;
  text-align:center;
  padding: 12px 0;
}}
</style>
"""

# Apply CSS early
st.set_page_config(page_title="FOOD VISION", page_icon="FoodVisionLogo.png", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ------------------ Data models ------------------
@dataclass
class UserProfile:
    user_id: str
    name: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    goal: Optional[str] = None
    daily_calories: Optional[int] = None


# ------------------ In-memory store helpers ------------------
# All reads now come from an in-memory store held in session_state.
# No reads are done from disk (no json file loads).

def load_store() -> dict:
    """Return the in-memory store (kept in st.session_state['_local_store'])."""
    return st.session_state.setdefault("_local_store", {"users": {}, "meals": {}, "undef_images": {}})


def save_store(obj: dict) -> None:
    """Overwrite the in-memory store. Does not touch disk."""
    st.session_state["_local_store"] = obj


def local_save_user(profile: UserProfile) -> None:
    s = load_store()
    s.setdefault("users", {})
    u = asdict(profile)
    u["daily_cal"] = profile.daily_calories
    s[profile.user_id] = s.get("users", {})
    # ensure we place under the proper key
    s = load_store()
    s.setdefault("users", {})
    s["users"][profile.user_id] = u
    save_store(s)


def local_get_user(uid: str) -> Optional[dict]:
    s = load_store()
    return s.get("users", {}).get(uid)


def local_save_meal(uid: str, food_name: str, calories: float, protein: float, fat: float, carbs: float, date_iso: str, thumb_path: str = "") -> None:
    s = load_store()
    s.setdefault("meals", {})
    s["meals"].setdefault(uid, []).append({
        "food_name": food_name,
        "calories": float(calories),
        "protein": float(protein),
        "fat": float(fat),
        "carbs": float(carbs),
        "date": date_iso,
        "thumb": thumb_path or "",
    })
    save_store(s)


def local_get_meals(uid: str) -> List[dict]:
    s = load_store()
    return s.get("meals", {}).get(uid, [])


def local_get_meals_by_date(uid: str, date_iso: str) -> List[dict]:
    return [m for m in local_get_meals(uid) if m.get("date") == date_iso]


def local_save_undef_image(uid: str, food_name: str, calories: float, protein: float, fat: float, carbs: float, date_iso: str, thumb_path: str = "", predicted_conf: Optional[float] = None, model_label: Optional[str] = None) -> None:
    s = load_store()
    s.setdefault("undef_images", {})
    s["undef_images"].setdefault(uid, []).append({
        "food_name": food_name,
        "calories": float(calories),
        "protein": float(protein),
        "fat": float(fat),
        "carbs": float(carbs),
        "date": date_iso,
        "thumb": thumb_path or "",
        "predicted_confidence": float(predicted_conf) if predicted_conf is not None else None,
        "model_label": model_label or "",
    })
    save_store(s)


# ------------------ Persistence wrappers (Firebase fallback) ------------------
def persist_user(profile: UserProfile) -> None:
    """Try Firebase first (if configured), otherwise store to in-memory store."""
    if USE_FIREBASE:
        try:
            save_user_data(profile.user_id, profile.name, profile.weight, profile.height, profile.age, profile.gender, profile.goal, profile.daily_calories)
            return
        except Exception as e:
            print(f"[persist_user] firebase error: {e}")
    local_save_user(profile)


def persist_meal(uid: str, food_name: str, calories: float, protein: float, fat: float, carbs: float, date_iso: str, thumb_path: Optional[str] = None) -> None:
    if USE_FIREBASE:
        try:
            save_meal(uid, food_name, calories, protein, fat, carbs, date_iso)
            return
        except Exception as e:
            print(f"[persist_meal] firebase error: {e}")
    local_save_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path or "")


def persist_undef_image(uid: str, food_name: str, calories: float, protein: float, fat: float, carbs: float, date_iso: str, thumb_path: Optional[str] = None, predicted_conf: Optional[float] = None, model_label: Optional[str] = None) -> None:
    if USE_FIREBASE:
        try:
            save_undef_image(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path=thumb_path, predicted_conf=predicted_conf, model_label=model_label)
            return
        except Exception as e:
            print(f"[persist_undef_image] firebase error: {e}")
    local_save_undef_image(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path or "", predicted_conf, model_label)


def fetch_user_profile(uid: str) -> Optional[dict]:
    """Get user profile from Firebase (if available) or in-memory store. Normalizes keys."""
    if not uid:
        return None
    if USE_FIREBASE:
        try:
            u = get_user_data(uid)
            if u:
                if "daily_calories" in u and u.get("daily_calories") is not None:
                    u["daily_cal"] = u.get("daily_calories")
                return u
        except Exception as e:
            print(f"[fetch_user_profile] firebase error: {e}")
    # fallback to in-memory store
    u = local_get_user(uid)
    if u:
        if "daily_calories" in u and "daily_cal" not in u:
            u["daily_cal"] = u.get("daily_calories")
    return u


def fetch_meals_by_date(uid: str, date_iso: str) -> List[dict]:
    if USE_FIREBASE:
        try:
            return get_meals_by_date(uid, date_iso)
        except Exception as e:
            print(f"[fetch_meals_by_date] firebase error: {e}")
    return local_get_meals_by_date(uid, date_iso)


def image_data_uri(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ------------------ Model & Prediction ------------------
MODEL = None
MODEL_READY = False
if YOLO_AVAILABLE and MODEL_PATH.exists():
    try:
        MODEL = YOLO(str(MODEL_PATH))
        MODEL_READY = True
    except Exception as e:
        print(f"[model] load error: {e}")
        MODEL_READY = False


def predict_food(image_path: str) -> Tuple[str, float]:
    if MODEL_READY and MODEL is not None:
        try:
            results = MODEL.predict(source=str(image_path), imgsz=640, conf=0.25, save=False)
            for r in results:
                if hasattr(r, "probs") and r.probs is not None:
                    idx = int(r.probs.top1)
                    name = r.names[idx]
                    conf = float(r.probs.top1conf)
                    return name.replace("_", " "), conf
                if r.boxes and len(r.boxes) > 0:
                    lbl = r.boxes.cls[0]
                    name = r.names[int(lbl)]
                    conf = float(r.boxes.conf[0])
                    return name.replace("_", " "), conf
        except Exception as e:
            print(f"[predict_food] model predict error: {e}")
            return "Unknown", 0.0
    options = [("Pad Thai", 0.9), ("Caesar Salad", 0.82), ("Burger", 0.88), ("Sushi", 0.8), ("Spring Rolls", 0.77)]
    idx = int(datetime.utcnow().timestamp()) % len(options)
    return options[idx]


# ------------------ Nutrition lookup ------------------
def fetch_nutrition(food_name: str) -> dict:
    if HAS_UTILS:
        try:
            info = get_nutrition(food_name)
            return {
                "calories": float(info.get("calories", 0)),
                "protein": float(info.get("protein", 0)),
                "fat": float(info.get("fat", 0)),
                "carbs": float(info.get("carbs", 0)),
            }
        except Exception as e:
            print(f"[fetch_nutrition] utils error: {e}")
    name = (food_name or "").lower()
    if "salad" in name:
        return {"calories": 220, "protein": 6, "fat": 14, "carbs": 15}
    if any(x in name for x in ("burger", "pizza")):
        return {"calories": 600, "protein": 25, "fat": 28, "carbs": 55}
    if "sushi" in name:
        return {"calories": 420, "protein": 18, "fat": 8, "carbs": 60}
    return {"calories": 350, "protein": 15, "fat": 12, "carbs": 40}


# ------------------ Utilities ------------------

def save_temp_thumbnail(uploaded_file, max_size: Tuple[int, int] = (300, 300)) -> Tuple[str, str]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    p = tmp.name
    try:
        im = Image.open(p)
        im.thumbnail(max_size)
        thumb_path = f"{p}.thumb.jpg"
        im.save(thumb_path, format="JPEG", quality=80)
        return p, thumb_path
    except Exception:
        return p, p


def safe_copy_thumb(src_thumb: Optional[str], uid: str) -> str:
    if not src_thumb:
        return ""
    try:
        dst = THUMBS_DIR / f"{uid}_{int(datetime.utcnow().timestamp())}.jpg"
        shutil.copy(src_thumb, dst)
        return str(dst)
    except Exception:
        return ""


# ------------------ Session state initialization ------------------

def ensure_session_state():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("daily_cal", None)
    st.session_state.setdefault("week_history", {})
    st.session_state.setdefault("page", "Profile")
    # in-memory store used instead of JSON file
    st.session_state.setdefault("_local_store", {"users": {}, "meals": {}, "undef_images": {}})


ensure_session_state()


# ------------------ UI components (cleaned) ------------------
LOGO_PATH = "FoodVisionLogo.png"
LOGO_AVAILABLE = Path(LOGO_PATH).exists()


def show_login() -> None:
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if LOGO_AVAILABLE:
            data_uri = image_data_uri(LOGO_PATH)
            if data_uri:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:center;align-items:center;margin-bottom:12px">
                    <img src="{data_uri}" width="140"
                        style="border-radius:12px;box-shadow:0 8px 20px rgba(0,0,0,0.12);"/>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(f"<h2 style='text-align:center;color:var(--white);margin-bottom:4px'>{APP_NAME}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:var(--accent);margin-top:0'>Your Smart Nutrition Assistant</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        email = st.text_input("Email", placeholder="you@example.com", key="login_email")
        pwd = st.text_input("Password", type="password", placeholder="Enter password", key="login_pwd")
        st.write("")
        if st.button("Sign in", key="btn_signin"):
            if not email or not pwd:
                st.error("Please enter both email and password.")
            else:
                uid = email.strip().lower()
                st.session_state.logged_in = True
                st.session_state.user_id = uid

                # Fetch profile from Firebase or in-memory store
                user = None
                if USE_FIREBASE:
                    try:
                        user = get_user_data(uid)
                    except Exception as e:
                        print(f"[login] get_user_data error: {e}")
                if not user:
                    user = local_get_user(uid)

                st.session_state.profile = user or {}
                daily = None
                if st.session_state.profile:
                    daily = st.session_state.profile.get("daily_calories") or st.session_state.profile.get("daily_cal")
                st.session_state.daily_cal = int(daily) if daily is not None else None

                st.session_state.page = "Dashboard" if st.session_state.profile else "Profile"
                st.success(f"Welcome, {email}!")
                st.rerun()


def sidebar_menu() -> str:
    with st.sidebar:
        st.markdown("<div style='padding:12px 0;'>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            if LOGO_AVAILABLE:
                st.image(LOGO_PATH, width=120)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color:var(--text);margin:6px 0'>{st.session_state.get('user_id','Guest')}</h4>", unsafe_allow_html=True)
        st.markdown("---")

        options = ["Profile", "Meal Analysis", "Dashboard", "Logout"]
        default_index = options.index(st.session_state.get("page", "Profile")) if st.session_state.get("page") in options else 0

        if HAS_OPTION_MENU:
            choice = option_menu(
                menu_title=None,
                options=options,
                icons=["person", "camera", "bar-chart", "box-arrow-right"],
                default_index=default_index,
                orientation="vertical",
                styles={
                    "container": {
                        "padding": "0!important",
                        "background-color": PRIMARY,
                    },
                    "icon": {"color": WHITE, "font-size": "18px"},
                    "nav-link": {
                        "font-size": "15px",
                        "text-align": "left",'color': WHITE,
                        "mfargin": "0px",
                        "--hover-color": "#2b3e53",
                    },
                    "nav-link-selected": {
                        "background-color": '#2b3e53',
                    },
                },
            )
        else:
            choice = st.radio("Navigate", options, index=default_index)

        st.markdown("---")
        st.markdown("<div style='font-size:12px;color:#ffffff;text-align:center'>Powered by FoodVision</div>", unsafe_allow_html=True)
    return choice


# ------------------ Pages ------------------

def page_profile() -> None:
    st.header("Profile")
    st.markdown("Enter a few details to calculate your daily calorie goal")

    uid = st.session_state.get("user_id")
    # prefer in-memory profile / firebase if available
    user = st.session_state.get("profile") or (get_user_data(uid) if USE_FIREBASE else None) or {}
    user = user or {}

    name_default = user.get("name", "")
    weight_default = float(user.get("weight", 70.0))
    height_default = float(user.get("height", 170.0))
    age_default = int(user.get("age", 30)) if user.get("age") is not None else 30
    gender_default = user.get("gender", "male")
    goal_default = user.get("goal", "maintain")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full name", value=name_default, placeholder="Optional", max_chars=40, key="p_name")
        weight = st.number_input("Weight (kg)", value=weight_default, min_value=30.0, max_value=200.0, step=0.5, key="p_weight")
        height = st.number_input("Height (cm)", value=height_default, min_value=120.0, max_value=220.0, step=1.0, key="p_height")
    with col2:
        age = st.number_input("Age", value=age_default, min_value=10, max_value=100, key="p_age")
        gender = st.selectbox("Gender", ["male", "female"], index=0 if gender_default == "male" else 1, key="p_gender")
        goal = st.selectbox("Goal", ["maintain", "lose", "gain"], index=["maintain", "lose", "gain"].index(goal_default), key="p_goal")

    st.write("")
    if st.button("Save Profile", key="save_profile"):
        if HAS_UTILS:
            daily = calculate_calories(weight, height, age, gender, "sedentary", goal)
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
            daily = int(bmr * 1.3)
            if goal == "lose":
                daily -= 500
            elif goal == "gain":
                daily += 500

        profile = UserProfile(user_id=uid or "anonymous", name=name, weight=weight, height=height, age=int(age), gender=gender, goal=goal, daily_calories=int(daily))
        persist_user(profile)

        st.session_state.profile = asdict(profile)
        st.session_state.daily_cal = int(daily)
        st.success(f"Profile saved — Daily goal: {int(daily)} kcal")
        st.session_state.page = "Dashboard"
        st.rerun()


def page_meal_analysis() -> None:
    st.header("Meal Analysis")
    st.markdown("Upload an image. Model will try to identify the food and fetch nutrition")

    uid = st.session_state.user_id or "anonymous"
    CONF_THRESHOLD = 0.3

    uploaded = st.file_uploader("Upload meal photo", type=["jpg", "jpeg", "png"], key="up_small")
    if uploaded:
        orig_path, thumb = save_temp_thumbnail(uploaded, max_size=(240, 240))
        st.image(thumb, width=180)
        st.write("")

        if st.button("Analyze", key="analyze_btn"):
            food_name, conf = predict_food(orig_path)
            st.session_state.pred_food = food_name
            st.session_state.pred_conf = float(conf)
            st.session_state.pred_orig_path = orig_path
            st.session_state.pred_thumb = thumb
            st.rerun()

    if st.session_state.get("pred_food") is not None:
        food_name = st.session_state.get("pred_food", "Unknown")
        conf = float(st.session_state.get("pred_conf", 0.0))
        pred_thumb = st.session_state.get("pred_thumb")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        iso = date.today().isoformat()

        if food_name != "Unknown" and conf >= CONF_THRESHOLD:
            st.markdown(f"**Detected:** {food_name}  —  *confidence {conf:.2%}*")
            nutrition = fetch_nutrition(food_name)
            st.markdown(f"**Calories:** {nutrition['calories']:.0f} kcal  |  **Protein:** {nutrition['protein']:.1f} g")
            st.markdown(f"**Fat:** {nutrition['fat']:.1f} g  |  **Carbs:** {nutrition['carbs']:.1f} g")
            st.write("")

            if st.button("Save detected", key="save_detected"):
                thumb_path = safe_copy_thumb(pred_thumb, uid)
                persist_meal(uid, food_name, nutrition["calories"], nutrition["protein"], nutrition["fat"], nutrition["carbs"], iso, thumb_path)
                st.session_state.week_history.setdefault(iso, 0)
                st.session_state.week_history[iso] += nutrition["calories"]
                st.success("Detected meal saved.")
                for k in ("pred_food", "pred_conf", "pred_orig_path", "pred_thumb"):
                    st.session_state.pop(k, None)
                st.rerun()

        else:
            st.markdown("**Model could not confidently identify this meal.** Please enter the food name and nutrition values below to save.")
            st.write("")
            manual_name = st.text_input("Food name (required to save)", value="", key="manual_name_input")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                cal_in = st.number_input("Calories (kcal)", min_value=0.0, value=0.0, step=1.0, key="manual_cal")
            with col_b:
                prot_in = st.number_input("Protein (g)", min_value=0.0, value=0.0, step=0.1, key="manual_prot")
            with col_c:
                fat_in = st.number_input("Fat (g)", min_value=0.0, value=0.0, step=0.1, key="manual_fat")
            with col_d:
                carbs_in = st.number_input("Carbs (g)", min_value=0.0, value=0.0, step=0.1, key="manual_carbs")

            st.write("")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Save manual", key="save_manual"):
                    if not manual_name.strip():
                        st.error("Please enter a food name to save.")
                    else:
                        thumb_path = safe_copy_thumb(pred_thumb, uid)
                        persist_undef_image(uid, manual_name.strip(), float(cal_in), float(prot_in), float(fat_in), float(carbs_in), iso, thumb_path=thumb_path, predicted_conf=conf, model_label=food_name)
                        persist_meal(uid, manual_name.strip(), float(cal_in), float(prot_in), float(fat_in), float(carbs_in), iso, thumb_path)
                        st.session_state.week_history.setdefault(iso, 0)
                        st.session_state.week_history[iso] += float(cal_in)
                        st.success("Manual meal saved to unknowns and added to your dashboard.")
                        for k in ("pred_food", "pred_conf", "pred_orig_path", "pred_thumb", "manual_name_input", "manual_cal", "manual_prot", "manual_fat", "manual_carbs"):
                            if k in st.session_state:
                                st.session_state.pop(k, None)
                        st.rerun()
            with col2:
                if st.button("Discard", key="discard_pred"):
                    for k in ("pred_food", "pred_conf", "pred_orig_path", "pred_thumb", "manual_name_input", "manual_cal", "manual_prot", "manual_fat", "manual_carbs"):
                        if k in st.session_state:
                            st.session_state.pop(k, None)
                    st.info("Prediction discarded.")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def page_dashboard() -> None:
    st.header("Dashboard")
    uid = st.session_state.user_id
    if not uid:
        st.info("Sign in and set up profile to see personalized dashboard.")
        return

    daily_goal = st.session_state.daily_cal or 2000
    today = date.today().isoformat()
    meals_today = fetch_meals_by_date(uid, today)
    df = pd.DataFrame(meals_today) if meals_today else pd.DataFrame(columns=["food_name", "calories", "protein", "fat", "carbs"])

    total_cal = df["calories"].sum() if not df.empty else 0
    total_p = df["protein"].sum() if not df.empty else 0
    total_f = df["fat"].sum() if not df.empty else 0
    total_c = df["carbs"].sum() if not df.empty else 0
    remaining = max(daily_goal - total_cal, 0)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card' style='background-color:var(--primary)'><h4 style='margin:0;color:{WHITE}'>Daily Goal</h4><h2 style='margin:6px;color:{WHITE}'>{int(daily_goal)} kcal</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card' style='background-color:var(--primary)'><h4 style='margin:0;color:{WHITE}'>Consumed</h4><h2 style='margin:6px;color:{WHITE}'>{int(total_cal)} kcal</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card' style='background-color:var(--primary)'><h4 style='margin:0;color:{WHITE}'>Remaining</h4><h2 style='margin:6px;color:{WHITE}'>{int(remaining)} kcal</h2></div>", unsafe_allow_html=True)

    st.markdown("")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Macro Split (today)")
        macro = pd.DataFrame({"nutrient": ["Protein", "Fat", "Carbs"], "grams": [total_p, total_f, total_c]})
        if macro["grams"].sum() == 0:
            st.info("No macro data for today.")
        else:
            figp = px.pie(macro, names="nutrient", values="grams", hole=0.4)
            st.plotly_chart(figp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Goal vs Consumed")
        bar = pd.DataFrame({"category": ["Goal", "Consumed"], "kcal": [daily_goal, total_cal]})
        figb = px.bar(bar, x="category", y="kcal", text="kcal")
        figb.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(figb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    tf_col1, tf_col2 = st.columns([1, 3])
    with tf_col1:
        timeframe = st.selectbox("Trend window", options=["7 days", "30 days", "90 days"], index=0)
    days_map = {"7 days": 7, "30 days": 30, "90 days": 90}
    days_n = days_map.get(timeframe, 7)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"{timeframe} Calories Trend")
    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(days_n - 1, -1, -1)]
    vals = []
    for d in days:
        if d in st.session_state.week_history:
            vals.append(st.session_state.week_history[d])
        else:
            rows = fetch_meals_by_date(uid, d)
            day_sum = sum([r.get("calories", 0) for r in rows]) if rows else 0
            vals.append(day_sum)
    trend_df = pd.DataFrame({"date": days, "kcal": vals})
    figline = px.line(trend_df, x="date", y="kcal", markers=True)
    if days_n > 14:
        figline.update_layout(xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(figline, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Daily Tip")
    tip = ""
    if total_p < max(20, total_cal * 0.08):
        tip = "Try adding a protein-rich snack (e.g., yogurt, eggs) to balance your intake."
    elif total_cal >= daily_goal:
        tip = "You've reached your goal today — consider lighter meals next."
    else:
        tip = "Great job — keep balanced meals and hydrate well!"
    st.info(tip)
    st.markdown("---")
    st.subheader("Meals Today")
    if df.empty:
        st.write("No meals logged today.")
    else:
        display_df = df[["food_name", "calories", "protein", "fat", "carbs"]].rename(columns={"food_name": "Food", "calories": "kcal", "protein": "Protein (g)", "fat": "Fat (g)", "carbs": "Carbs (g)"})
        st.dataframe(display_df, height=220)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ Footer ------------------

def show_footer() -> None:
    st.markdown("<footer>Powered by FoodVision | Smart Nutrition Assistant 🍏</footer>", unsafe_allow_html=True)


# ------------------ Main app ------------------

def main():
    ensure_session_state()
    if not st.session_state.logged_in:
        show_login()
        show_footer()
        return

    choice = sidebar_menu()

    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.user_id = ""
        st.rerun()
        return

    if choice == "Profile":
        page_profile()
    elif choice == "Meal Analysis":
        page_meal_analysis()
    elif choice == "Dashboard":
        page_dashboard()
    else:
        st.info("Select a page from the sidebar.")

    show_footer()


if __name__ == "__main__":
    main()
