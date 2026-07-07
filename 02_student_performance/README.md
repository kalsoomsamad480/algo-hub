# 🎓 Student Performance Prediction

Predicting whether a student will **Pass** or **Fail** using historical academic, demographic, and social data — built as a Week 2 classification project.

## 📌 Project Overview

This project uses the **UCI Student Performance Dataset** to predict student outcomes using two classification models: **Logistic Regression** and **Random Forest**. The pipeline covers data preprocessing, feature selection, model training, evaluation, and deployment as an interactive **Streamlit** web app.

## 🎯 Objectives

- Predict student performance (Pass/Fail) using historical academic data
- Compare Logistic Regression vs Random Forest for classification
- Apply feature selection to identify the most predictive factors
- Visualize results and feature importance
- Deploy an interactive prediction tool with Streamlit

## 🧠 Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Non-linear ensemble classifier, also used for feature importance |

## 📊 Dataset Information

- **Source**: [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **File used**: `student-mat.csv` (Math course subset)
- **Rows**: 395 students
- **Columns**: 33 (30 features + G1, G2, G3 grades)
- **Target**: Derived `pass` column — `1` if final grade (G3) ≥ 10, else `0`

**Feature categories:**
- Demographics: age, sex, address, family size
- Family background: parents' education & jobs, family relationship quality
- Academic history: study time, past failures, G1/G2 grades, absences
- Social/lifestyle: free time, going out, alcohol consumption, health

## ⚙️ Installation Guide

**1. Clone the repository**
```bash
git clone https://github.com/kalsoomsamad480/student-performance-prediction.git
cd student-performance-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**
Download `student-mat.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/320/student+performance) and place it in the project folder (same level as `preprocessing.py`).

**4. Run the pipeline in order**
```bash
python preprocessing.py
python training.py
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 🗂️ Project Structure

```
student-performance-prediction/
├── preprocessing.py       # Data cleaning, encoding, feature selection
├── training.py            # Model training & evaluation
├── app.py                 # Streamlit web app
├── requirements.txt        # Python dependencies
├── screenshots/            # App screenshots
└── README.md
```

## 📈 Results & Screenshots

| Model | Accuracy |
|---|---|
| Logistic Regression | 86.08% |
| Random Forest | 87.34% |

*(Fill in your actual accuracy numbers from `models/model_comparison.csv` after running training.py)*

**App Home / Input Form**

![App Home](screenshots/app_home.png)

**Prediction Result**

![Prediction Result](screenshots/prediction_result.png)

## 🛠️ Tech Stack

- Python 3.11
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Streamlit

## 👤 Author

**Kalsoom Samad**
- GitHub: [@kalsoomsamad480](https://github.com/kalsoomsamad480)
- LinkedIn: [Kalsoom Samad](https://linkedin.com/in/kalsoom-samad-ba468139a)

Part of the **Algo Hub** project portfolio.