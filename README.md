# 🥗 FoodVision

<p align="center">
  <img src="assets/logo.png" alt="FoodVision Logo" width="180"/>
</p>

**FoodVision** is an AI-powered nutrition analysis platform that helps users understand and manage their diet simply by taking a photo of their meal.  
It identifies the food, extracts key nutritional values (calories, carbs, protein, fat), and provides personalized calorie goals and insights — making healthy eating simple, visual, and data-driven.

---

## 🚀 Features

- 📸 **Meal Recognition:** Upload a meal photo and get instant detection using a custom-trained **YOLOv11-small** model.  
- 🍎 **Nutrition Estimation:** Automatically retrieves nutritional data (calories, macros) from the **USDA API**.  
- 📊 **Interactive Dashboards:** Visualize daily intake, trends, and macronutrient breakdowns.  
- 🎯 **Personalized Goals:** Calculates calorie targets based on user height, weight, age, and dietary goal (maintain, lose, or gain weight).  
- 💬 **User Feedback:** Allows tracking of undetected meals and continuous model improvement.  
- 🔐 **User Profiles:** Each user can log in, view their stats, and track progress through an intuitive interface.

---

## 🧠 Model Overview

- **Dataset:** [Food101](https://www.kaggle.com/datasets/dansbecker/food-101)  
  – 101,000 images (1,000 per class)  
- **Model:** YOLOv11-small (classification) with 5.5M parameters  
- **Baseline Accuracy:** 66%  
- **Final Model Accuracy:** **89%** (after hyperparameter tuning and early stopping at 75 epochs)  
- **Inference Speed:** 14.4 ms per image  

---

## 📊 Dashboards

### Meal Analysis Dashboard
<p align="center">
  <img src="assets/meal_analysis.png" alt="Meal Analysis Dashboard" width="80%"/>
</p>

Shows detected food items, calories, macros, and daily goal comparisons.

### User Profile Dashboard
<p align="center">
  <img src="assets/profile_dashboard.png" alt="User Profile Dashboard" width="80%"/>
</p>

Displays personalized calorie targets, history, and AI-based eating tips.

### Data & Activity Dashboard
<p align="center">
  <img src="assets/dashboard.png" alt="Data Dashboard" width="80%"/>
</p>

Provides overall performance analytics, user engagement, and nutrition tracking statistics.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Model** | YOLOv11-small |
| **Dataset** | Food101 |
| **Backend** | Python, FastAPI |
| **Frontend** | Streamlit |
| **Database/API** | Firebase, USDA API |
| **Visualization** | Plotly, Matplotlib |

---

## 🧩 Project Architecture

