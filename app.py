import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(
    page_title="Employee Salary & Appraisal Predictor",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Salary & Appraisal Predictor (INR)")
st.markdown("""
This Machine Learning dashboard estimates compensation packages and appraisal benchmarks 
using a **Random Forest Regressor** trained on corporate industry salary scales.
""")

# 2. Synthetic Benchmark Dataset & Model Training
@st.cache_resource
def train_salary_model():
    np.random.seed(42)
    n_samples = 1500
    
    # Generate realistic industry career features
    experience = np.random.uniform(0.5, 20.0, n_samples)
    role_tier = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.35, 0.2, 0.1]) # 1: Support/Ops, 2: Analyst/Dev, 3: Senior/Lead, 4: Architect/Manager
    education = np.random.choice([1, 2, 3], size=n_samples, p=[0.55, 0.35, 0.10]) # 1: Bachelor's, 2: Master's, 3: Ph.D/Tier-1
    certifications = np.random.randint(0, 6, size=n_samples)
    perf_rating = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.15, 0.50, 0.20, 0.10]) # Appraisal rating (1-5)

    # Base formula with realistic non-linear market multipliers + noise
    base_salary = (
        3.5 + 
        (experience * 1.4) + 
        (role_tier * 3.8) + 
        (education * 2.2) + 
        (certifications * 0.8) + 
        (perf_rating * 1.1) + 
        np.random.normal(0, 1.2, n_samples)
    )
    # Ensure salary stays above baseline
    base_salary = np.clip(base_salary, 3.0, 60.0)

    X = pd.DataFrame({
        'experience': experience,
        'role_tier': role_tier,
        'education': education,
        'certifications': certifications,
        'perf_rating': perf_rating
    })
    y = base_salary

    model = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

model = train_salary_model()

# 3. Sidebar Inputs
st.sidebar.header("📋 Candidate Profile")

role_map = {
    "Operations / Associate Support": 1,
    "Software Engineer / Data Analyst": 2,
    "Senior Consultant / Technical Lead": 3,
    "Engineering Manager / Solution Architect": 4
}
selected_role = st.sidebar.selectbox("Job Role Tier", list(role_map.keys()), index=1)
role_val = role_map[selected_role]

exp_val = st.sidebar.slider("Total Relevant Experience (Years)", 0.0, 20.0, 3.5, 0.5)

edu_map = {
    "Bachelor's Degree (B.Tech / B.Sc / BCA / B.Com)": 1,
    "Master's Degree (M.Tech / MBA / MCA / M.Sc)": 2,
    "Doctorate (Ph.D.) or Tier-1 Institute Alumni": 3
}
selected_edu = st.sidebar.selectbox("Education Level", list(edu_map.keys()), index=0)
edu_val = edu_map[selected_edu]

cert_val = st.sidebar.slider("Number of Industry Certifications (AWS, GCP, PMP, etc.)", 0, 5, 2)
perf_val = st.sidebar.slider("Latest Annual Performance Rating (1: Low to 5: Outstanding)", 1, 5, 4)

# Feature DataFrame for Inference
input_features = pd.DataFrame([{
    'experience': exp_val,
    'role_tier': role_val,
    'education': edu_val,
    'certifications': cert_val,
    'perf_rating': perf_val
}])

# 4. Display & Output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Profile Overview")
    profile_df = pd.DataFrame({
        "Attribute": ["Experience", "Designation Tier", "Education", "Certifications", "Appraisal Rating"],
        "Value": [f"{exp_val} Years", selected_role, selected_edu.split('(')[0].strip(), f"{cert_val}", f"{perf_val} / 5"]
    })
    st.table(profile_df)

with col2:
    st.subheader("Compensation Assessment")
    if st.button("Evaluate Compensation", type="primary"):
        predicted_lpa = model.predict(input_features)[0]
        monthly_takehome = (predicted_lpa * 100000) / 12
        
        st.success(f"### Estimated CTC: **₹{predicted_lpa:.2f} Lakhs / annum (LPA)**")
        st.metric(label="Estimated Gross Monthly", value=f"₹{monthly_takehome:,.0f} / mo")
        
        # Suggested Appraisal Bracket
        if perf_val >= 4:
            st.info("🌟 **Appraisal Recommendation:** High-performer track with a recommended **15% – 25%** compensation adjustment.")
        elif perf_val == 3:
            st.info("📊 **Appraisal Recommendation:** Standard merit track with a recommended **8% – 12%** adjustment.")
        else:
            st.warning("⚠️ **Appraisal Recommendation:** Performance improvement plan or baseline adjustment.")
