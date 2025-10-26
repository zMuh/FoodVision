import streamlit as st
import tempfile
import shutil
import datetime
from ultralytics import YOLO
from app.utils import get_nutrition, calculate_calories
from app.firebase_utils import save_user_data, save_meal, get_meals_by_date

# --- Load YOLO model ---
MODEL_PATH = "./models/best.pt"
model = YOLO(MODEL_PATH)

st.title("🍽️ Smart Nutrition Assistant")
st.write("Enter your personal data and upload a meal image to analyze it 🔍")

tab1, tab2, tab3 = st.tabs(["👤 User Profile", "📸 Meal Analysis", "📊 Dashboard"])

# =========================
# 👤 USER PROFILE
# =========================
with tab1:
    st.subheader("Enter your basic information")
    user_id = st.text_input("🆔 User ID (name or email)")
    name = st.text_input("Full Name")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["male", "female"])
    activity = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active", "very_active"])
    goal = st.selectbox("Goal", ["maintain", "lose", "gain"])

    if st.button("💾 Calculate & Save Profile"):
        daily_cal = calculate_calories(weight, height, age, gender, activity, goal)
        save_user_data(user_id, name, weight, height, goal, daily_cal)
        st.session_state["daily_cal"] = daily_cal
        st.session_state["user_id"] = user_id
        st.success(f"✅ Profile saved successfully! Your recommended daily calories: {daily_cal:.0f} kcal.")

# =========================
# 🍔 MEAL ANALYSIS
# =========================
with tab2:
    st.subheader("Upload your meal image")
    user_id_input = st.text_input("🆔 Enter your User ID again:")
    uploaded_file = st.file_uploader("📤 Choose a meal image", type=["jpg", "jpeg", "png"])

    if uploaded_file and user_id_input:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(uploaded_file, tmp)
            tmp_path = tmp.name

        results = model.predict(source=tmp_path, imgsz=640, conf=0.25, save=False)

        # Extract predicted class and confidence
        food_name, confidence = "Unknown", 0.0
        for r in results:
            if hasattr(r, "probs"):
                top_idx = r.probs.top1
                food_name = r.names[top_idx]
                confidence = float(r.probs.top1conf)
                break

        # If prediction uncertain
        if food_name == "Unknown" or confidence < 0.5:
            st.warning("⚠️ The model could not confidently identify the meal. Please enter it manually:")
            manual_name = st.text_input("🍽️ Enter meal name manually:")
            if manual_name:
                food_name_clean = manual_name
                nutrition_info = get_nutrition(food_name_clean)
            else:
                nutrition_info = {}
        else:
            food_name_clean = food_name.replace("_", " ")
            nutrition_info = get_nutrition(food_name_clean)

        # Display results
        st.image(uploaded_file, caption=f"{food_name_clean} ({confidence:.2f})", use_container_width=True)
        if nutrition_info:
            st.markdown("### 🍽️ Nutritional Information")
            st.json(nutrition_info)

            # Save to Firebase
            date_today = datetime.date.today().isoformat()
            save_meal(user_id_input, food_name_clean, nutrition_info['calories'],
                      nutrition_info['protein'], nutrition_info['fat'],
                      nutrition_info['carbs'], date_today)
            st.session_state["last_meal"] = nutrition_info
            st.success("✅ Meal information saved successfully!")

        # Display meals for today
        st.markdown("---")
        st.subheader("📊 Your Meals Today")
        meals_today = get_meals_by_date(user_id_input, datetime.date.today().isoformat())
        if meals_today:
            st.dataframe(meals_today)
        else:
            st.info("No meals recorded for today yet.")

        # Cleanup
        try:
            shutil.rmtree(tmp_path)
        except Exception:
            pass

# =========================
# 📊 DASHBOARD TAB
# =========================
with tab3:
    st.subheader("📈 Nutrition Dashboard")
    if "daily_cal" not in st.session_state or "last_meal" not in st.session_state:
        st.warning("⚠️ Please enter your profile and analyze at least one meal first.")
    else:
        daily_goal = st.session_state["daily_cal"]
        last_meal = st.session_state["last_meal"]

        st.markdown(f"### 🎯 Daily Calorie Goal: **{daily_goal:.0f} kcal**")
        st.markdown("### 🍱 Last Meal Nutrition")
        st.json(last_meal)

        consumed = last_meal['calories']
        remaining = daily_goal - consumed

        st.metric(label="Calories Consumed", value=f"{consumed} kcal")
        st.metric(label="Calories Remaining", value=f"{remaining if remaining > 0 else 0} kcal")

        if remaining <= 0:
            st.success("✅ You've reached your daily calorie goal!")
        elif remaining < daily_goal * 0.2:
            st.warning("⚠️ You're close to your daily limit.")
        else:
            st.info("💪 Keep going — you're within your daily plan!")

        import pandas as pd
        import plotly.express as px

        st.markdown("---")
        st.markdown("### 🍩 Macronutrient Breakdown")

        macro_data = pd.DataFrame({
            'Nutrient': ['Protein', 'Fat', 'Carbs'],
            'Grams': [
                last_meal.get('protein', 0),
                last_meal.get('fat', 0),
                last_meal.get('carbs', 0)
            ]
        })

        fig = px.pie(
            macro_data,
            values='Grams',
            names='Nutrient',
            title='Macronutrient Composition of Last Meal',
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Daily Calories vs Meals")

        if "user_id" in st.session_state:
            today_date = datetime.date.today().isoformat()
            meals_today = get_meals_by_date(st.session_state["user_id"], today_date)

            if meals_today:
                df_meals = pd.DataFrame(meals_today)
                total_calories = df_meals["calories"].sum()

                bar_data = pd.DataFrame({
                    "Category": ["Daily Goal", "Calories Consumed"],
                    "Calories": [daily_goal, total_calories]
                })

                fig2 = px.bar(
                    bar_data,
                    x="Category",
                    y="Calories",
                    text="Calories",
                    color="Category",
                    title="Daily Goal vs Today's Meals",
                )
                fig2.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig2.update_layout(yaxis_title="Calories (kcal)", xaxis_title=None)
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown(f"**Total Calories Consumed Today:** {total_calories:.0f} kcal / {daily_goal:.0f} kcal")

                # >>> ADDED MEALS DETAIL GRAPH <<<
                st.markdown("---")
                st.markdown("### 🍽️ Calories per Meal Today")

                fig3 = px.bar(
                    df_meals,
                    x="food_name",
                    y="calories",
                    text="calories",
                    color="food_name",
                    title="Calories Breakdown per Meal",
                )
                fig3.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig3.update_layout(yaxis_title="Calories (kcal)", xaxis_title="Meal")
                st.plotly_chart(fig3, use_container_width=True)

                for i, row in df_meals.iterrows():
                    st.markdown(
                        f"- 🍴 **{row['food_name']}** → {row['calories']} kcal "
                        f"({row['protein']}g protein, {row['fat']}g fat, {row['carbs']}g carbs)"
                    )
                # >>> END OF ADDED MEALS DETAIL GRAPH <<<

            else:
                st.info("🍽️ No meals recorded for today yet.")
        else:
            st.warning("⚠️ Please enter your profile first.")