
 #app.py
#"""
#Smart Nutrition Assistant - Clean & Simple Streamlit App (single file)
#Features:
#- Landing / Login (simple local email+password)
#- Sidebar menu with logo and 3 pages: Profile, Meal Analysis, Dashboard
#- Small compact inputs and small image uploader (thumbnail)
#- Dashboard is a compact grid with cards + charts
#- Uses local JSON storage fallback; will use app.utils and app.firebase_utils if available
#- Uses ultralytics YOLO if available and models/best.pt exists; otherwise uses mock predictions
#- Colors / visual identity based on provided palette
#"""
#
#import streamlit as st
#from pathlib import Path
#from datetime import date, timedelta, datetime
#import tempfile, shutil, json, os
#from PIL import Image
#import pandas as pd
#import plotly.express as px
#import numpy as np
#
## ----------------------------
## Try optional libs / helpers
## ----------------------------
#try:
#    from streamlit_option_menu import option_menu
#    HAS_OPTION_MENU = True
#except Exception:
#    HAS_OPTION_MENU = False
#
#try:
#    from ultralytics import YOLO
#    YOLO_AVAILABLE = True
#except Exception:
#    YOLO_AVAILABLE = False
#
#USE_FIREBASE = False
#try:
#    from app.firebase_utils import save_user_data, save_meal, get_meals_by_date
#    USE_FIREBASE = True
#except Exception:
#    USE_FIREBASE = False
#
#HAS_UTILS = False
#try:
#    from app.utils import get_nutrition, calculate_calories
#    HAS_UTILS = True
#except Exception:
#    HAS_UTILS = False
#
## ----------------------------
## Colors / visuals
## ----------------------------
#PRIMARY = "#69bba4"
#PRIMARY_DARK = "#5fbc7a"
#ACCENT = "#7a838d"
#NAV_BG = "#1c2d3f"
#BG = "#1b2d42"
#TEXT = "#f7f9fb"
#CARD_BG = "#ffffff"
#
## ----------------------------
## Page config & CSS
## ----------------------------
#st.set_page_config(page_title="FOOD VISION", page_icon="FoodVisionLogo.png", layout="wide")
#
#st.markdown(
#    f"""
#    <style>
#      /* page */
#      .app-header {{ display:flex; align-items:center; gap:14px; }}
#      .logo-img {{ width:72px; height:72px; border-radius:14px; }}
#      .title-main {{ font-size:22px; font-weight:800; color:{NAV_BG}; margin:0; }}
#      .subtitle {{ color:{ACCENT}; margin:0; font-size:13px; }}
#      /* sidebar */
#      .stSidebar .sidebar-content {{
#        background: linear-gradient(180deg, {NAV_BG}, {BG});
#        color: {TEXT};
#      }}
#      /* small inputs */
#      input[type="text"], input[type="password"], input[type="number"] {{
#        height:34px;
#      }}
#      .small-button > button {{ padding:6px 12px; border-radius:8px; background:{PRIMARY}; color:white; font-weight:600; }}
#      /* cards */
#      .card {{ background:{CARD_BG}; border-radius:12px; padding:14px; box-shadow: 0 6px 18px rgba(27,45,66,0.06); }}
#      .muted {{ color:#6b7280; font-size:13px; }}
#      footer {{ text-align:center; color:#94a3b8; padding:18px 0; }}
#    </style>
#    """,
#    unsafe_allow_html=True,
#)
#
## ----------------------------
## Local storage fallback
## ----------------------------
#STORE_PATH = Path("sna_store.json")
#if not STORE_PATH.exists():
#    STORE_PATH.write_text(json.dumps({"users": {}, "meals": {}}, indent=2))
#
#def load_store():
#    return json.loads(STORE_PATH.read_text())
#
#def save_store(obj):
#    STORE_PATH.write_text(json.dumps(obj, indent=2, default=str))
#
#def local_save_user(uid, name, weight, height, goal, daily_cal):
#    s = load_store()
#    s["users"][uid] = {"name": name, "weight": weight, "height": height, "goal": goal, "daily_cal": daily_cal}
#    save_store(s)
#
#def local_save_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path=None):
#    s = load_store()
#    s["meals"].setdefault(uid, []).append({
#        "food_name": food_name,
#        "calories": float(calories),
#        "protein": float(protein),
#        "fat": float(fat),
#        "carbs": float(carbs),
#        "date": date_iso,
#        "thumb": thumb_path or ""
#    })
#    save_store(s)
#
#def local_get_meals(uid):
#    s = load_store()
#    return s["meals"].get(uid, [])
#
#def local_get_meals_by_date(uid, date_iso):
#    return [m for m in local_get_meals(uid) if m["date"] == date_iso]
#
## wrappers (try firebase first)
#def persist_user(uid, name, weight, height, goal, daily_cal):
#    if USE_FIREBASE:
#        try:
#            save_user_data(uid, name, weight, height, goal, daily_cal)
#            return
#        except Exception:
#            pass
#    local_save_user(uid, name, weight, height, goal, daily_cal)
#
#def persist_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path=None):
#    if USE_FIREBASE:
#        try:
#            save_meal(uid, food_name, calories, protein, fat, carbs, date_iso)
#            return
#        except Exception:
#            pass
#    local_save_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path)
#
#def fetch_meals_by_date(uid, date_iso):
#    if USE_FIREBASE:
#        try:
#            return get_meals_by_date(uid, date_iso)
#        except Exception:
#            pass
#    return local_get_meals_by_date(uid, date_iso)
#
## ----------------------------
## YOLO or mock predict
## ----------------------------
#MODEL_PATH = Path("models/best.pt")
#MODEL = None
#MODEL_READY = False
#if YOLO_AVAILABLE and MODEL_PATH.exists():
#    try:
#        MODEL = YOLO(str(MODEL_PATH))
#        MODEL_READY = True
#    except Exception:
#        MODEL_READY = False
#
#def predict_food(image_path):
#    if MODEL_READY:
#        try:
#            results = MODEL.predict(source=str(image_path), imgsz=640, conf=0.25, save=False)
#            food_name, conf = "Unknown", 0.0
#            for r in results:
#                if hasattr(r, "probs"):
#                    idx = r.probs.top1
#                    food_name = r.names[idx]
#                    conf = float(r.probs.top1conf)
#                    break
#            return food_name.replace("_", " "), conf
#        except Exception:
#            return "Unknown", 0.0
#    # mock
#    options = [("Pad Thai",0.9),("Caesar Salad",0.82),("Burger",0.88),("Sushi",0.8),("Spring Rolls",0.77)]
#    idx = int(datetime.utcnow().timestamp()) % len(options)
#    return options[idx]
#
## ----------------------------
## Nutrition fetch (real or heuristic)
## ----------------------------
#def fetch_nutrition(food_name):
#    if HAS_UTILS:
#        try:
#            info = get_nutrition(food_name)
#            return {
#                "calories": float(info.get("calories",0)),
#                "protein": float(info.get("protein",0)),
#                "fat": float(info.get("fat",0)),
#                "carbs": float(info.get("carbs",0))
#            }
#        except Exception:
#            pass
#    # heuristic fallbacks
#    name = (food_name or "").lower()
#    if "salad" in name:
#        return {"calories":220,"protein":6,"fat":14,"carbs":15}
#    if "burger" in name or "pizza" in name:
#        return {"calories":600,"protein":25,"fat":28,"carbs":55}
#    if "sushi" in name:
#        return {"calories":420,"protein":18,"fat":8,"carbs":60}
#    # default
#    return {"calories":350,"protein":15,"fat":12,"carbs":40}
#
## ----------------------------
## Session init
## ----------------------------
#if "logged_in" not in st.session_state:
#    st.session_state.logged_in = False
#if "user_id" not in st.session_state:
#    st.session_state.user_id = ""
#if "daily_cal" not in st.session_state:
#    st.session_state.daily_cal = None
#if "week_history" not in st.session_state:
#    st.session_state.week_history = {}  # date_iso -> calories
#
## ----------------------------
## Logo path (provided)
## ----------------------------
#LOGO_PATH = "FoodVisionLogo.png"
#LOGO_AVAILABLE = Path(LOGO_PATH).exists()
#
## ----------------------------
## Helper: compact small uploader preview
## ----------------------------
#def save_temp_thumbnail(uploaded_file, max_size=(300,300)):
#    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#    tmp.write(uploaded_file.getvalue())
#    tmp.flush()
#    tmp.close()
#    p = tmp.name
#    try:
#        im = Image.open(p)
#        im.thumbnail(max_size)
#        thumb_path = p + ".thumb.jpg"
#        im.save(thumb_path, format="JPEG", quality=80)
#        return p, thumb_path
#    except Exception:
#        return p, p
#
## ----------------------------
## Landing / Login screen
## ----------------------------
#def show_login():
#    st.markdown("<div style='display:flex;justify-content:center;margin-top:18px'>", unsafe_allow_html=True)
#    if LOGO_AVAILABLE:
#        st.image(LOGO_PATH, width=96)
#    st.markdown("</div>", unsafe_allow_html=True)
#
#    st.markdown("<h2 style='text-align:center;color:#0f172a;margin-bottom:4px'>Smart Nutrition Assistant</h2>", unsafe_allow_html=True)
#    st.markdown(f"<p style='text-align:center;color:{ACCENT};margin-top:0'>Your AI-powered meal analyzer</p>", unsafe_allow_html=True)
#    st.write("")
#    # compact inputs: use columns to reduce width
#    c1, c2, c3 = st.columns([1,2,1])
#    with c2:
#        email = st.text_input("Email", placeholder="you@example.com", key="login_email")
#        pwd = st.text_input("Password", type="password", placeholder="Enter password", key="login_pwd")
#        st.write("")
#        if st.button("Sign in", key="btn_signin"):
#            if not email or not pwd:
#                st.error("Please enter both email and password.")
#            else:
#                # simple local signin: store user_id once
#                st.session_state.logged_in = True
#                st.session_state.user_id = email.lower()
#                st.success(f"Welcome, {email}!")
#                # if user saved in local store, load daily_cal to session
#                store = load_store()
#                user = store.get("users", {}).get(st.session_state.user_id)
#                if user:
#                    st.session_state.daily_cal = user.get("daily_cal")
#                st.experimental_rerun()
#
## ----------------------------
## Sidebar menu
## ----------------------------
#def sidebar_menu():
#    st.sidebar.markdown("<div style='display:flex;flex-direction:column;align-items:center;padding:8px'>", unsafe_allow_html=True)
#    if LOGO_AVAILABLE:
#        st.sidebar.image(LOGO_PATH, width=72)
#    st.sidebar.markdown(f"<h4 style='color:{TEXT};margin:6px 0'>{st.session_state.user_id}</h4>", unsafe_allow_html=True)
#    st.sidebar.markdown("---")
#    if HAS_OPTION_MENU:
#        choice = option_menu(None, ["Profile","Meal Analysis","Dashboard","Logout"],
#                             icons=["person","camera","bar-chart","box-arrow-right"],
#                             menu_icon="cast", default_index=0, orientation="vertical",
#                             styles={"nav-link":{"font-size":"14px","text-align":"left","padding":"8px 12px"}})
#    else:
#        choice = st.sidebar.radio("Navigate", ["Profile","Meal Analysis","Dashboard","Logout"])
#    st.sidebar.markdown("---")
#    st.sidebar.markdown("<div style='font-size:12px;color:#94a3b8'>Powered by FoodVision</div>", unsafe_allow_html=True)
#    return choice
#
## ----------------------------
## Profile page (compact inputs)
## ----------------------------
#def page_profile():
#    st.header("Profile")
#    st.markdown("Enter a few details to calculate your daily calorie goal")
#    col1, col2 = st.columns(2)
#    with col1:
#        name = st.text_input("Full name", placeholder="Optional", key="p_name", max_chars=40)
#        weight = st.number_input("Weight (kg)", value=70.0, min_value=30.0, max_value=200.0, step=0.5, key="p_weight")
#        height = st.number_input("Height (cm)", value=170.0, min_value=120.0, max_value=220.0, step=1.0, key="p_height")
#    with col2:
#        age = st.number_input("Age", value=30, min_value=10, max_value=100, key="p_age")
#        gender = st.selectbox("Gender", ["male","female"], index=0, key="p_gender")
#        goal = st.selectbox("Goal", ["maintain","lose","gain"], index=0, key="p_goal")
#    st.write("")
#    if st.button("Save Profile", key="save_profile"):
#        if HAS_UTILS:
#            daily = calculate_calories(weight, height, age, gender, activity_level="sedentary", goal=goal)
#        else:
#            # Mifflin fallback
#            if gender == "male":
#                bmr = 10*weight + 6.25*height - 5*age + 5
#            else:
#                bmr = 10*weight + 6.25*height - 5*age - 161
#            daily = int(bmr * 1.3)
#            if goal == "lose": daily -= 500
#            elif goal == "gain": daily += 500
#        persist_user(st.session_state.user_id, name, weight, height, goal, int(daily))
#        st.session_state.daily_cal = int(daily)
#        st.success(f"Profile saved — Daily goal: {int(daily)} kcal")
#
## ----------------------------
## Meal Analysis page (compact uploader)
## ----------------------------
#def page_meal_analysis():
#    st.header("Meal Analysis")
#    st.markdown("Upload a small image (thumbnail preview). Model will try to identify the food and fetch nutrition")
#    uid = st.session_state.user_id
#    c1, c2 = st.columns([2,1])
#    with c1:
#        uploaded = st.file_uploader("Small image (try < 1MB)", type=["jpg","jpeg","png"], key="up_small")
#    with c2:
#        st.markdown("Quick tips:")
#        st.markdown("- Use clear top-down photos")
#        st.markdown("- Crop close to the food")
#    if uploaded:
#        orig_path, thumb = save_temp_thumbnail(uploaded, max_size=(240,240))
#        st.image(thumb, width=180)
#        st.write("")
#        if st.button("Analyze & Save", key="analyze_save"):
#            food_name, conf = predict_food(orig_path)
#            if (food_name=="Unknown" or conf < 0.5):
#                # prompt input
#                manual = st.text_input("Model unsure — enter food name manually", key="manual_name")
#                if manual:
#                    food_name = manual
#            nutrition = fetch_nutrition(food_name)
#            # show compact card
#            st.markdown("<div class='card'>", unsafe_allow_html=True)
#            st.markdown(f"**Detected:** {food_name}  —  *confidence {conf:.2f}*")
#            st.markdown(f"**Calories:** {nutrition['calories']:.0f} kcal  |  **Protein:** {nutrition['protein']:.1f} g")
#            st.markdown(f"**Fat:** {nutrition['fat']:.1f} g  |  **Carbs:** {nutrition['carbs']:.1f} g")
#            st.markdown("</div>", unsafe_allow_html=True)
#            # persist
#            iso = date.today().isoformat()
#            # save thumbnail into store folder for simple reference
#            thumbs_dir = Path("thumbs")
#            thumbs_dir.mkdir(exist_ok=True)
#            dst = thumbs_dir / f"{uid}_{int(datetime.utcnow().timestamp())}.jpg"
#            try:
#                shutil.copy(thumb, dst)
#                thumb_path = str(dst)
#            except Exception:
#                thumb_path = ""
#            persist_meal(uid, food_name, nutrition['calories'], nutrition['protein'], nutrition['fat'], nutrition['carbs'], iso, thumb_path)
#            # update session week history
#            st.session_state.week_history.setdefault(iso, 0)
#            st.session_state.week_history[iso] += nutrition["calories"]
#            st.success("Meal saved.")
#
## ----------------------------
## Dashboard (compact grid)
## ----------------------------
#def page_dashboard():
#    st.header("Dashboard")
#    uid = st.session_state.user_id
#    if not uid:
#        st.info("Sign in and set up profile to see personalized dashboard.")
#        return
#    daily_goal = st.session_state.daily_cal or 2000
#    today = date.today().isoformat()
#    meals_today = fetch_meals_by_date(uid, today)
#    df = pd.DataFrame(meals_today) if meals_today else pd.DataFrame(columns=["food_name","calories","protein","fat","carbs"])
#
#    total_cal = df["calories"].sum() if not df.empty else 0
#    total_p = df["protein"].sum() if not df.empty else 0
#    total_f = df["fat"].sum() if not df.empty else 0
#    total_c = df["carbs"].sum() if not df.empty else 0
#    remaining = max(daily_goal - total_cal, 0)
#
#    # Top cards row (3 cards)
#    col1, col2, col3 = st.columns(3)
#    col1.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Daily Goal</h4><h2 style='margin:6px 0'>{int(daily_goal)} kcal</h2></div>", unsafe_allow_html=True)
#    col2.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Consumed</h4><h2 style='margin:6px 0'>{int(total_cal)} kcal</h2></div>", unsafe_allow_html=True)
#    col3.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Remaining</h4><h2 style='margin:6px 0'>{int(remaining)} kcal</h2></div>", unsafe_allow_html=True)
#
#    st.markdown("")  # small gap
#
#    # Grid: left pie + right bar
#    left, right = st.columns([1,1])
#    with left:
#        st.markdown("<div class='card'>", unsafe_allow_html=True)
#        st.subheader("Macro Split (today)")
#        macro = pd.DataFrame({"nutrient":["Protein","Fat","Carbs"], "grams":[total_p, total_f, total_c]})
#        if macro[["grams"]].sum().values[0] == 0:
#            st.info("No macro data for today.")
#        else:
#            figp = px.pie(macro, names="nutrient", values="grams", hole=0.4, color_discrete_sequence=[PRIMARY, PRIMARY_DARK, ACCENT])
#            st.plotly_chart(figp, use_container_width=True)
#        st.markdown("</div>", unsafe_allow_html=True)
#    with right:
#        st.markdown("<div class='card'>", unsafe_allow_html=True)
#        st.subheader("Goal vs Consumed")
#        bar = pd.DataFrame({"category":["Goal","Consumed"], "kcal":[daily_goal, total_cal]})
#        figb = px.bar(bar, x="category", y="kcal", text="kcal", color="category", color_discrete_sequence=[PRIMARY, PRIMARY_DARK])
#        figb.update_traces(texttemplate='%{text:.0f}', textposition='outside')
#        st.plotly_chart(figb, use_container_width=True)
#        st.markdown("</div>", unsafe_allow_html=True)
#
#    st.markdown("")  # gap
#
#    # 7-day line trend
#    st.markdown("<div class='card'>", unsafe_allow_html=True)
#    st.subheader("7-day Calories Trend")
#    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
#    vals = []
#    for d in days:
#        if d in st.session_state.week_history:
#            vals.append(st.session_state.week_history[d])
#        else:
#            rows = fetch_meals_by_date(uid, d)
#            vals.append(sum([r.get("calories",0) for r in rows]) if rows else 0)
#    trend_df = pd.DataFrame({"date": days, "kcal": vals})
#    figline = px.line(trend_df, x="date", y="kcal", markers=True)
#    st.plotly_chart(figline, use_container_width=True)
#    st.markdown("</div>", unsafe_allow_html=True)
#
#    st.markdown("")  # gap
#
#    # Tips & meal list
#    st.markdown("<div class='card'>", unsafe_allow_html=True)
#    st.subheader("Daily Tip")
#    tip = ""
#    if total_p < max(20, total_cal * 0.08):
#        tip = "Try adding a protein-rich snack (e.g., yogurt, eggs) to balance your intake."
#    elif total_cal >= daily_goal:
#        tip = "You've reached your goal today — consider lighter meals next."
#    else:
#        tip = "Great job — keep balanced meals and hydrate well!"
#    st.info(tip)
#    st.markdown("---")
#    st.subheader("Meals Today")
#    if df.empty:
#        st.write("No meals logged today.")
#    else:
#        display_df = df[["food_name","calories","protein","fat","carbs"]].rename(columns={
#            "food_name":"Food", "calories":"kcal","protein":"Protein (g)","fat":"Fat (g)","carbs":"Carbs (g)"
#        })
#        st.dataframe(display_df, height=220)
#    st.markdown("</div>", unsafe_allow_html=True)
#
## ----------------------------
## Footer
## ----------------------------
#def show_footer():
#    st.markdown("<footer>Powered by FoodVision | Smart Nutrition Assistant 🍏</footer>", unsafe_allow_html=True)
#
## ----------------------------
## Main app flow
## ----------------------------
#def main():
#    if not st.session_state.logged_in:
#        show_login()
#        show_footer()
#        return
#
#    choice = sidebar_menu()
#    if choice == "Logout":
#        st.session_state.logged_in = False
#        st.session_state.user_id = ""
#        st.experimental_rerun()
#        return
#
#    # show top small header with logo and app name
#    header_cols = st.columns([0.08, 1])
#    with header_cols[0]:
#        if LOGO_AVAILABLE:
#            st.image(LOGO_PATH, width=48)
#    with header_cols[1]:
#        st.markdown("<div style='display:flex;flex-direction:column;justify-content:center'>"
#                    f"<strong style='color:#0f172a'>Smart Nutrition Assistant</strong>"
#                    f"<span style='color:{ACCENT};font-size:12px'>Your compact meal analyzer</span>"
#                    "</div>", unsafe_allow_html=True)
#
#    # route pages
#    if choice == "Profile":
#        page_profile()
#    elif choice == "Meal Analysis":
#        page_meal_analysis()
#    elif choice == "Dashboard":
#        page_dashboard()
#    else:
#        st.write("Select a page from the sidebar.")
#
#    show_footer()
#
#if __name__ == "__main__":
#     main()
#









## app.py
"""
Smart Nutrition Assistant - Clean & Simple Streamlit App (single file)
Features:
- Landing / Login (simple local email+password)
- Sidebar menu with logo and 3 pages: Profile, Meal Analysis, Dashboard
- Small compact inputs and small image uploader (thumbnail)
- Dashboard is a compact grid with cards + charts
- Uses local JSON storage fallback; will use app.utils and app.firebase_utils if available
- Uses ultralytics YOLO if available and models/best.pt exists; otherwise uses mock predictions
- Colors / visual identity based on provided palette
"""

import streamlit as st
from pathlib import Path
from datetime import date, timedelta, datetime
import tempfile, shutil, json, os
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------------------
# Try optional libs / helpers
# ----------------------------
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

USE_FIREBASE = False
try:
    from app.firebase_utils import save_user_data, save_meal, get_meals_by_date
    USE_FIREBASE = True
except Exception:
    USE_FIREBASE = False

HAS_UTILS = False
try:
    from app.utils import get_nutrition, calculate_calories
    HAS_UTILS = True
except Exception:
    HAS_UTILS = False

# ----------------------------
# Colors / visuals
# ----------------------------
PRIMARY = "#69bba4"
PRIMARY_DARK = "#5fbc7a"
ACCENT = "#7a838d"
NAV_BG = "#1c2d3f"
BG = "#1b2d42"
TEXT = "#f7f9fb"
CARD_BG = "#1b2d42b3"

# ----------------------------
# Page config & CSS
# ----------------------------
st.set_page_config(page_title="FOOD VISION", page_icon="FoodVisionLogo.png", layout="wide")

st.markdown(
    f"""
    <style>
      /* page */
      .app-header {{ display:flex; justify-content:center;align-items:center; gap:14px; }}
      .logo-img {{ width:72px; height:72px; border-radius:14px; }}
      .title-main {{ font-size:22px; font-weight:800; color:{NAV_BG}; margin:0; }}
      .subtitle {{ color:{ACCENT}; margin:0; font-size:13px; }}
      /* sidebar */
      .stSidebar .sidebar-content {{
        background: linear-gradient(180deg, {NAV_BG}, {BG});
        color: {TEXT};
      }}
      /* small inputs */
      input[type="text"], input[type="password"], input[type="number"] {{
        height:34px;
      }}
      .small-button > button {{ padding:6px 12px; border-radius:8px; background:{PRIMARY}; color:white; font-weight:600; }}
      /* cards */
      .card {{ background:{CARD_BG}; border-radius:12px; padding:14px; box-shadow: 0 6px 18px rgba(27,45,66,0.06); }}
      .muted {{ color:#6b7280; font-size:13px; }}
      footer {{ text-align:center; color:#94a3b8; padding:18px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Local storage fallback
# ----------------------------
STORE_PATH = Path("sna_store.json")
if not STORE_PATH.exists():
    STORE_PATH.write_text(json.dumps({"users": {}, "meals": {}}, indent=2))

def load_store():
    return json.loads(STORE_PATH.read_text())

def save_store(obj):
    STORE_PATH.write_text(json.dumps(obj, indent=2, default=str))

def local_save_user(uid, name, weight, height, goal, daily_cal):
    s = load_store()
    s["users"][uid] = {"name": name, "weight": weight, "height": height, "goal": goal, "daily_cal": daily_cal}
    save_store(s)

def local_save_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path=None):
    s = load_store()
    s["meals"].setdefault(uid, []).append({
        "food_name": food_name,
        "calories": float(calories),
        "protein": float(protein),
        "fat": float(fat),
        "carbs": float(carbs),
        "date": date_iso,
        "thumb": thumb_path or ""
    })
    save_store(s)

def local_get_meals(uid):
    s = load_store()
    return s["meals"].get(uid, [])

def local_get_meals_by_date(uid, date_iso):
    return [m for m in local_get_meals(uid) if m["date"] == date_iso]

# wrappers (try firebase first)
def persist_user(uid, name, weight, height, goal, daily_cal):
    if USE_FIREBASE:
        try:
            save_user_data(uid, name, weight, height, goal, daily_cal)
            return
        except Exception:
            pass
    local_save_user(uid, name, weight, height, goal, daily_cal)

def persist_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path=None):
    if USE_FIREBASE:
        try:
            save_meal(uid, food_name, calories, protein, fat, carbs, date_iso)
            return
        except Exception:
            pass
    local_save_meal(uid, food_name, calories, protein, fat, carbs, date_iso, thumb_path)

def fetch_meals_by_date(uid, date_iso):
    if USE_FIREBASE:
        try:
            return get_meals_by_date(uid, date_iso)
        except Exception:
            pass
    return local_get_meals_by_date(uid, date_iso)

# ----------------------------
# YOLO or mock predict
# ----------------------------
MODEL_PATH = Path("models/best.pt")
MODEL = None
MODEL_READY = False
if YOLO_AVAILABLE and MODEL_PATH.exists():
    try:
        MODEL = YOLO(str(MODEL_PATH))
        MODEL_READY = True
    except Exception:
        MODEL_READY = False

def predict_food(image_path):
    if MODEL_READY:
        try:
            results = MODEL.predict(source=str(image_path), imgsz=640, conf=0.25, save=False)
            food_name, conf = "Unknown", 0.0
            for r in results:
                if hasattr(r, "probs"):
                    idx = r.probs.top1
                    food_name = r.names[idx]
                    conf = float(r.probs.top1conf)
                    break
            return food_name.replace("_", " "), conf
        except Exception:
            return "Unknown", 0.0
    # mock
    options = [("Pad Thai",0.9),("Caesar Salad",0.82),("Burger",0.88),("Sushi",0.8),("Spring Rolls",0.77)]
    idx = int(datetime.utcnow().timestamp()) % len(options)
    return options[idx]

# ----------------------------
# Nutrition fetch (real or heuristic)
# ----------------------------
def fetch_nutrition(food_name):
    if HAS_UTILS:
        try:
            info = get_nutrition(food_name)
            return {
                "calories": float(info.get("calories",0)),
                "protein": float(info.get("protein",0)),
                "fat": float(info.get("fat",0)),
                "carbs": float(info.get("carbs",0))
            }
        except Exception:
            pass
    # heuristic fallbacks
    name = (food_name or "").lower()
    if "salad" in name:
        return {"calories":220,"protein":6,"fat":14,"carbs":15}
    if "burger" in name or "pizza" in name:
        return {"calories":600,"protein":25,"fat":28,"carbs":55}
    if "sushi" in name:
        return {"calories":420,"protein":18,"fat":8,"carbs":60}
    # default
    return {"calories":350,"protein":15,"fat":12,"carbs":40}

# ----------------------------
# Session init
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "daily_cal" not in st.session_state:
    st.session_state.daily_cal = None
if "week_history" not in st.session_state:
    st.session_state.week_history = {}  # date_iso -> calories

# ----------------------------
# Logo path (provided)
# ----------------------------
LOGO_PATH = "FoodVisionLogo.png"
LOGO_AVAILABLE = Path(LOGO_PATH).exists()

# ----------------------------
# Helper: compact small uploader preview
# ----------------------------
def save_temp_thumbnail(uploaded_file, max_size=(300,300)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    p = tmp.name
    try:
        im = Image.open(p)
        im.thumbnail(max_size)
        thumb_path = p + ".thumb.jpg"
        im.save(thumb_path, format="JPEG", quality=80)
        return p, thumb_path
    except Exception:
        return p, p

# ----------------------------
# Landing / Login screen
# ----------------------------
def show_login():
    st.markdown("<div style='display:flex;justify-content:center;margin-top:36px'>", unsafe_allow_html=True)
    if LOGO_AVAILABLE:
        st.image(LOGO_PATH, width=160)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;color:#0f172a;margin-bottom:4px'>FOOD VISION</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{ACCENT};margin-top:0'>Your Smart Nutrition Assistant</p>", unsafe_allow_html=True)
    st.write("")
    # compact inputs: use columns to reduce width
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        email = st.text_input("Email", placeholder="you@example.com", key="login_email")
        pwd = st.text_input("Password", type="password", placeholder="Enter password", key="login_pwd")
        st.write("")
        if st.button("Sign in", key="btn_signin"):
            if not email or not pwd:
                st.error("Please enter both email and password.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_id = email.lower()
                st.success(f"Welcome, {email}!")
                # load daily_cal if exists
                store = load_store()
                user = store.get("users", {}).get(st.session_state.user_id)
                if user:
                    st.session_state.daily_cal = user.get("daily_cal")
                return  # بدل st.experimental_rerun()

# ----------------------------
# Sidebar menu
# ----------------------------
def sidebar_menu():
    st.sidebar.markdown("<div style='display:flex;flex-direction:column;align-items:center;padding:8px'>", unsafe_allow_html=True)
    if LOGO_AVAILABLE:
        st.sidebar.image(LOGO_PATH, width=72)
    st.sidebar.markdown(f"<h4 style='color:{BG};margin:6px 0'>{st.session_state.user_id}</h4>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    if HAS_OPTION_MENU:
        choice = option_menu(None, ["Profile","Meal Analysis","Dashboard","Logout"],
                             icons=["person","camera","bar-chart","box-arrow-right"],
                             menu_icon="cast", default_index=0, orientation="vertical",
                             styles={"nav-link":{"font-size":"14px","text-align":"left","padding":"8px 12px"}})
    else:
        choice = st.sidebar.radio("Navigate", ["Profile","Meal Analysis","Dashboard","Logout"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:12px;color:#94a3b8'>Powered by FoodVision</div>", unsafe_allow_html=True)
    return choice

# ----------------------------
# Profile page
# ----------------------------
def page_profile():
    st.header("Profile")
    st.markdown("Enter a few details to calculate your daily calorie goal")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full name", placeholder="Optional", key="p_name", max_chars=40)
        weight = st.number_input("Weight (kg)", value=70.0, min_value=30.0, max_value=200.0, step=0.5, key="p_weight")
        height = st.number_input("Height (cm)", value=170.0, min_value=120.0, max_value=220.0, step=1.0, key="p_height")
    with col2:
        age = st.number_input("Age", value=30, min_value=10, max_value=100, key="p_age")
        gender = st.selectbox("Gender", ["male","female"], index=0, key="p_gender")
        goal = st.selectbox("Goal", ["maintain","lose","gain"], index=0, key="p_goal")
    st.write("")
    if st.button("Save Profile", key="save_profile"):
        if HAS_UTILS:
            daily = calculate_calories(weight, height, age, gender, activity_level="sedentary", goal=goal)
        else:
            if gender == "male":
                bmr = 10*weight + 6.25*height - 5*age + 5
            else:
                bmr = 10*weight + 6.25*height - 5*age - 161
            daily = int(bmr * 1.3)
            if goal == "lose": daily -= 500
            elif goal == "gain": daily += 500
        persist_user(st.session_state.user_id, name, weight, height, goal, int(daily))
        st.session_state.daily_cal = int(daily)
        st.success(f"Profile saved — Daily goal: {int(daily)} kcal")

# ----------------------------
# Meal Analysis page
# ----------------------------
def page_meal_analysis():
    st.header("Meal Analysis")
    st.markdown("Upload an image. Model will try to identify the food and fetch nutrition")
    uid = st.session_state.user_id
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], key="up_small")  # حذف النص القديم

    if uploaded:
        orig_path, thumb = save_temp_thumbnail(uploaded, max_size=(240,240))
        st.image(thumb, width=180)
        st.write("")
        if st.button("Analyze & Save", key="analyze_save"):
            food_name, conf = predict_food(orig_path)
            if (food_name=="Unknown" or conf < 0.5):
                manual = st.text_input("Model unsure — enter food name manually", key="manual_name")
                if manual:
                    food_name = manual
            nutrition = fetch_nutrition(food_name)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Detected:** {food_name}  —  *confidence {conf:.2f}*")
            st.markdown(f"**Calories:** {nutrition['calories']:.0f} kcal  |  **Protein:** {nutrition['protein']:.1f} g")
            st.markdown(f"**Fat:** {nutrition['fat']:.1f} g  |  **Carbs:** {nutrition['carbs']:.1f} g")
            st.markdown("</div>", unsafe_allow_html=True)
            iso = date.today().isoformat()
            thumbs_dir = Path("thumbs")
            thumbs_dir.mkdir(exist_ok=True)
            dst = thumbs_dir / f"{uid}_{int(datetime.utcnow().timestamp())}.jpg"
            try:
                shutil.copy(thumb, dst)
                thumb_path = str(dst)
            except Exception:
                thumb_path = ""
            persist_meal(uid, food_name, nutrition['calories'], nutrition['protein'], nutrition['fat'], nutrition['carbs'], iso, thumb_path)
            st.session_state.week_history.setdefault(iso, 0)
            st.session_state.week_history[iso] += nutrition["calories"]
            st.success("Meal saved.")

# ----------------------------
# Dashboard
# ----------------------------

def page_dashboard():
    st.header("Dashboard")
    uid = st.session_state.user_id
    if not uid:
        st.info("Sign in and set up profile to see personalized dashboard.")
        return
    daily_goal = st.session_state.daily_cal or 2000
    today = date.today().isoformat()
    meals_today = fetch_meals_by_date(uid, today)
    df = pd.DataFrame(meals_today) if meals_today else pd.DataFrame(columns=["food_name","calories","protein","fat","carbs"])

    total_cal = df["calories"].sum() if not df.empty else 0
    total_p = df["protein"].sum() if not df.empty else 0
    total_f = df["fat"].sum() if not df.empty else 0
    total_c = df["carbs"].sum() if not df.empty else 0
    remaining = max(daily_goal - total_cal, 0)

    # Top cards row (3 cards)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Daily Goal</h4><h2 style='margin:6px 0'>{int(daily_goal)} kcal</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Consumed</h4><h2 style='margin:6px 0'>{int(total_cal)} kcal</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4 style='margin:0;color:{NAV_BG}'>Remaining</h4><h2 style='margin:6px 0'>{int(remaining)} kcal</h2></div>", unsafe_allow_html=True)

    st.markdown("")  # small gap

    # Grid: left pie + right bar
    left, right = st.columns([1,1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Macro Split (today)")
        macro = pd.DataFrame({"nutrient":["Protein","Fat","Carbs"], "grams":[total_p, total_f, total_c]})
        if macro[["grams"]].sum().values[0] == 0:
            st.info("No macro data for today.")
        else:
            figp = px.pie(macro, names="nutrient", values="grams", hole=0.4, color_discrete_sequence=[PRIMARY, PRIMARY_DARK, ACCENT])
            st.plotly_chart(figp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Goal vs Consumed")
        bar = pd.DataFrame({"category":["Goal","Consumed"], "kcal":[daily_goal, total_cal]})
        figb = px.bar(bar, x="category", y="kcal", text="kcal", color="category", color_discrete_sequence=[PRIMARY, PRIMARY_DARK])
        figb.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(figb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # gap

    # 7-day line trend
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("7-day Calories Trend")
    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(6,-1,-1)]
    vals = []
    for d in days:
        if d in st.session_state.week_history:
            vals.append(st.session_state.week_history[d])
        else:
            rows = fetch_meals_by_date(uid, d)
            vals.append(sum([r.get("calories",0) for r in rows]) if rows else 0)
    trend_df = pd.DataFrame({"date": days, "kcal": vals})
    figline = px.line(trend_df, x="date", y="kcal", markers=True)
    st.plotly_chart(figline, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # gap

    # Tips & meal list
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
        display_df = df[["food_name","calories","protein","fat","carbs"]].rename(columns={
            "food_name":"Food", "calories":"kcal","protein":"Protein (g)","fat":"Fat (g)","carbs":"Carbs (g)"
        })
        st.dataframe(display_df, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
def show_footer():
    st.markdown("<footer>Powered by FoodVision | Smart Nutrition Assistant 🍏</footer>", unsafe_allow_html=True)

# ----------------------------
# Main
# ----------------------------
def main():
    if not st.session_state.logged_in:
        show_login()
        return
    choice = sidebar_menu()
    if choice=="Profile":
        page_profile()
    elif choice=="Meal Analysis":
        page_meal_analysis()
    elif choice=="Dashboard":
        page_dashboard()
    elif choice=="Log out":
        st.session_state.logged_in = False
        st.session_state.user_id = ""
        return

    st.markdown("<footer>FoodVision &copy; 2025</footer>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
