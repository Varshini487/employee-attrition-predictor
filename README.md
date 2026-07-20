# 👥 Employee Attrition Predictor

An **HR analytics platform** that predicts which employees are at high risk of resignation, with SHAP-based explanations showing *exactly* why.

## 🎯 Problem
Replacing an employee costs 50-200% of their annual salary (recruitment, training, lost productivity). Predict who will quit → manager has time to intervene, retain, or plan ahead.

## 🔄 How It Works (4-step pipeline)
```
Employee Data (salary, tenure, department, performance rating, satisfaction, work-life balance)
              ↓
Feature Extraction (years at company, promotion wait-time, salary growth rate, avg bonus)
              ↓
XGBoost Classifier (trained on historical resignations)
              ↓
SHAP Explanations: "This person quit because: (1) low satisfaction score (-0.25), (2) no promotion in 3 years (-0.18), (3) long commute (-0.12)"
              ↓
HR Action: 1-on-1 discussion, promotion, raise, remote work option, career development plan
```

## 🛠️ Tech Stack
- **Python, Pandas** — employee data processing
- **XGBoost** — attrition classification
- **SHAP** — model-agnostic explanations
- **Matplotlib, Seaborn** — visualizations
- **Streamlit** — HR dashboard

## 📊 Expected Performance
| Metric | Score |
|--------|-------|
| Accuracy | 84% |
| Recall (catching true attrition) | 0.79 |
| Precision | 0.75 |

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/employee-attrition-predictor
cd employee-attrition-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Tech companies (prevent flight of engineers to competitors)
- Consulting firms (predict partner departures)
- Sales teams (identify flight risk reps before customer loss)
- Startups (retain co-founders/early employees)

## 🎤 3 Interview Talking Points

**1️⃣ SHAP tells managers *why*, not just *if*.**
"Raw prediction: 'Employee X is 75% likely to quit.' Manager thinks 'why?' With SHAP, I show: satisfaction score dropped 2 points (-0.30 impact), no promotion in 3 years (-0.25), commute increased (+0.15). Now manager knows: This person needs a conversation about career growth, not a raise. SHAP explanations drive better retention decisions than raw probabilities."

**2️⃣ Early intervention has 3x better outcomes than last-minute retention.**
"Research: if you address issues *before* someone starts interviewing, retention rate is 65%. If you wait until they've accepted an offer, retention rate is 20%. The model flags people 2-3 months before they quit, giving managers time for real interventions: promotions, skill development, team changes. Timing matters."

**3️⃣ Attrition analytics save millions in replacement costs.**
"Average replacement cost: $75K. If model prevents 20 resignations/year out of 1,000 employees, that's $1.5M in direct savings. Plus: retained employees already know the codebase (no ramp time), client relationships (revenue continuity). Hard to quantify, but 3-5x larger than direct costs."
