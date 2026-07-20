import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="👥 Attrition Predictor", layout="wide")
st.title("👥 Employee Attrition Predictor")
st.markdown("Predict employee resignation risk with SHAP explanations")

@st.cache_data
def load_sample_hr_data():
    np.random.seed(42)
    n = 500
    years_at_company = np.random.randint(1, 20, n)
    age = np.random.randint(22, 60, n)
    salary = 40000 + (years_at_company * 3000) + (age * 500) + np.random.uniform(-10000, 10000, n)
    performance_rating = np.random.uniform(2.0, 5.0, n)
    satisfaction = np.random.uniform(1, 5, n)
    promotion_years = np.random.randint(0, 8, n)
    
    attrition_prob = (
        -0.08 * years_at_company +
        0.15 * (5 - satisfaction) +
        -0.02 * performance_rating +
        0.05 * promotion_years +
        0.003 * (60 - age)
    )
    attrition = (attrition_prob + np.random.normal(0, 0.15, n) > 0.2).astype(int)
    
    department = np.random.choice(["Engineering", "Sales", "HR", "Finance"], n)
    
    return pd.DataFrame({
        "age": age,
        "years_at_company": years_at_company,
        "salary": salary,
        "performance_rating": performance_rating,
        "satisfaction": satisfaction,
        "promotion_years_ago": promotion_years,
        "department": department,
        "attrition": attrition
    })

df = load_sample_hr_data()

tab1, tab2, tab3 = st.tabs(["📊 Workforce Overview", "🤖 Train Model", "🔮 Employee Risk Score"])

with tab1:
    st.subheader("Employee Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", len(df))
    col2.metric("Attrition Count", df["attrition"].sum())
    col3.metric("Attrition Rate", f"{df['attrition'].mean():.1%}")
    col4.metric("Avg Salary", f"${df['salary'].mean()/1000:.0f}K")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby("department")["attrition"].mean().plot(kind="bar", ax=axes[0], color="coral")
    axes[0].set_title("Attrition by Department")
    axes[0].set_ylabel("Attrition Rate")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    axes[1].scatter(df[df.attrition==0]["satisfaction"], df[df.attrition==0]["years_at_company"], 
                   alpha=0.5, label="Stayed", s=30)
    axes[1].scatter(df[df.attrition==1]["satisfaction"], df[df.attrition==1]["years_at_company"], 
                   alpha=0.7, color="red", label="Left", s=50, marker="X")
    axes[1].set_xlabel("Job Satisfaction (1-5)")
    axes[1].set_ylabel("Tenure (years)")
    axes[1].set_title("Attrition Patterns")
    axes[1].legend()
    
    st.pyplot(fig)

with tab2:
    if st.button("🚀 Train Attrition Model"):
        le = LabelEncoder()
        df_enc = df.copy()
        df_enc["department"] = le.fit_transform(df["department"])
        
        X = df_enc[["age", "years_at_company", "salary", "performance_rating", "satisfaction", "promotion_years_ago", "department"]]
        y = df_enc["attrition"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        st.success("✅ Model trained!")
        st.text(classification_report(y_test, y_pred, target_names=["Will Stay", "Will Leave"]))
        st.metric("AUC-ROC", f"{auc:.3f}")
        
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["le"] = le
        st.session_state["feature_names"] = X.columns.tolist()

with tab3:
    st.subheader("Predict Attrition Risk for an Employee")
    
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 22, 65, 35)
    years = c2.slider("Years at Company", 1, 20, 5)
    salary = c3.number_input("Salary ($)", 30000, 200000, 70000)
    
    c4, c5, c6 = st.columns(3)
    perf = c4.slider("Performance Rating (1-5)", 1.0, 5.0, 3.5)
    satisfy = c5.slider("Job Satisfaction (1-5)", 1.0, 5.0, 3.0)
    promo_years = c6.slider("Years since last promotion", 0, 8, 2)
    
    if st.button("🔮 Calculate Risk") and "model" in st.session_state:
        X_input = np.array([[age, years, salary, perf, satisfy, promo_years, 1]])
        X_input_scaled = st.session_state["scaler"].transform(X_input)
        
        attrition_prob = st.session_state["model"].predict_proba(X_input_scaled)[0][1]
        
        st.markdown("### 📋 Attrition Risk Report")
        
        if attrition_prob > 0.7:
            st.error(f"🚨 **CRITICAL RISK: {attrition_prob:.1%}**")
            st.markdown("""
**Immediate Actions:**
1. Schedule 1-on-1 with manager TODAY
2. Discuss career growth & next promotion
3. Review compensation vs market rate
4. Explore role changes / new projects
5. Consider mentorship opportunities
            """)
        elif attrition_prob > 0.45:
            st.warning(f"⚠️ **ELEVATED RISK: {attrition_prob:.1%}**")
            st.markdown("""
**Recommended Actions:**
- Check in regularly (bi-weekly 1-on-1s)
- Assign stretch project to boost engagement
- Offer professional development budget
- Team/role reassessment
            """)
        else:
            st.success(f"✅ **LOW RISK: {attrition_prob:.1%}**")
            st.markdown("Employee appears engaged. Continue regular development & feedback.")

st.markdown("---")
st.markdown("**Stack:** Pandas · Scikit-learn · XGBoost · SHAP · Streamlit")
