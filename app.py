
from pathlib import Path
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression #matches model type

st.set_page_config(page_title="Weight Loss Tool", page_icon="⚖️")

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.pkl"
DATA_INFO_PATH = HERE / "data_info.pkl"

# ---------- Load artifacts ----------
@st.cache_resource
def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

try:
    best_model = load_pickle(MODEL_PATH) #load model from model.pkl
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n{e}")
    st.stop()

try:
    data_info = load_pickle(DATA_INFO_PATH) #load data_info from data_info.pkl
except Exception as e:
    st.error(
        f"Could not load data_info at {DATA_INFO_PATH}.\n"
        f"Ensure data_info.pkl exists and includes expected_columns.\n{e}"
    )
    st.stop()

#extract items from data_info dictionary
expected_features = data_info["expected_features"]
categorical_unique_vals = data_info["categorical_unique_vals"]
numerical_ranges = data_info["numerical_ranges"]
numeric_features = data_info["numeric_features"]

experience_levels = ["Beginner", "Intermediate", "Advanced"] #special encoding for label-encoded variable
experience_ord = { 
    "Beginner": 1,
    "Intermediate": 2,
    "Advanced": 3
}

# Helper: label->code for UI selections just for Experience_Level
def label_to_code(selection_label: str, mapping: dict) -> str:
    # mapping is code->label; invert to label->code
    inv = {v: k for k, v in mapping.items()}
    return inv[selection_label]

st.title("Weight Loss Tool")
st.caption("Two steps for machine learning to guide weight loss")

st.header("Step 1: Enter Exercise Information to Estimate Calorie Burn")

def num_slider(name, default, lo, hi, step=1):
    r = numerical_ranges.get(name, {})
    lo = int(r.get("min", lo))
    hi = int(r.get("max", hi))
    val = int(r.get("default", default))
    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

def num_slider_float(name, default, lo, hi, step=1):
    r = numerical_ranges.get(name, {})
    lo = float(r.get("min", lo))
    hi = float(r.get("max", hi))
    val = float(r.get("default", default))
    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

# all sliders for numeric features as they will appear in Streamlit
age = num_slider("Age", 2, 1, 3)
weight = num_slider_float("Weight (kg)", 2, 1, 3, step=0.01)
height = num_slider_float("Height (m)", 2, 1, 3, step=0.01)
max_bpm = num_slider("Max_BPM", 2, 1, 3)
avg_bpm = num_slider("Avg_BPM", 2, 1, 3)
resting_bpm = num_slider("Resting_BPM", 2, 1, 3)
duration = num_slider_float("Session_Duration (hours)", 2, 1, 3, step=0.1)
st.caption("*Fat percentage is a whole number out of 100")
fat_percentage = num_slider("Fat_Percentage", 2, 1, 3)
st.caption("*Water intake is a daily average")
water_intake = num_slider_float("Water_Intake (liters)", 2, 1, 3, step=0.01)
frequency = num_slider("Workout_Frequency (days/week)", 2, 1, 3)
bmi = num_slider_float("BMI", 2, 1, 3, step=0.1)

#all dropdowns for categorical features as they will appear in Streamlit
gender = st.selectbox("Gender", categorical_unique_vals["Gender"])
workout_type = st.selectbox("Workout Type", categorical_unique_vals["Workout_Type"])
experience_label = st.selectbox("Experience Level", experience_levels)
experience = experience_ord[experience_label]

new_user = {
    "Age": age,
    "Gender": gender,
    "Weight (kg)": weight,
    "Height (m)": height,
    "Max_BPM": max_bpm,
    "Avg_BPM": avg_bpm,
    "Resting_BPM": resting_bpm,
    "Session_Duration (hours)": duration,
    "Workout_Type": workout_type,
    "Fat_Percentage": fat_percentage,
    "Water_Intake (liters)": water_intake,
    "Workout_Frequency (days/week)": frequency,
    "Experience_Level": experience,
    "BMI": bmi
}

new_user_df = pd.DataFrame([new_user])

st.divider()
if st.button("Predict"):
    try:
        pred = best_model.predict(new_user_df)[0]

        st.subheader("Prediction Result")
        if pred:
            st.success(np.round(pred, decimals=2))
        else:
            st.error("Prediction Error")

    except Exception as e:
        st.error(f"Inference failed: {e}")
