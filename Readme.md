# 🐦 Iris Classification Web Service

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Flask](https://img.shields.io/badge/Flask-1.1.2-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Project Overview
An end-to-end machine learning pipeline demonstrating:

- 🔍 **Exploration & Feature Engineering** on the Iris dataset
- 🏋️ **Training** of three models: Logistic Regression, Decision Tree & Random Forest
- 💾 **Serialization** of pipelines into pickle files
- 🌐 **Deployment** via Flask: both a **RESTful JSON API** and a **responsive web interface**
- 📬 **Consumption** through a web form and Postman, validating predictions and latency

---

## 📊 Dataset
The **Iris** dataset (150 samples) with 3 species (50 each of *setosa*, *versicolor*, *virginica*). Features:

| Feature               | Type   | Description                    |
|:----------------------|:-------|:-------------------------------|
| sepal length (cm)     | float  | Sepal length                   |
| sepal width (cm)      | float  | Sepal width                    |
| petal length (cm)     | float  | Petal length                   |
| petal width (cm)      | float  | Petal width                    |
| target (0,1,2)        | int    | Encoded species class          |

---

## 🔍 EDA & Feature Engineering
1. **Summary Stats & Validation**: No missing values; feature ranges checked via `df.describe()`.
2. **Pairplot Visualization**: Showed clear setosa cluster and overlap between other species.
3. **Correlation Heatmap**: Petal length & width highly correlated (≈0.96).
4. **Engineered Features**:

   | Feature             | Formula                           |
   |:--------------------|:----------------------------------|
   | Sepal Area          | sepal length × sepal width        |
   | Petal Area          | petal length × petal width        |
   | Sepal/Petal Ratio   | sepal length ÷ petal length       |

---

## 🏋️ Model Training
All models use a **scaling + classifier** pipeline on 7 numeric features (4 raw + 3 engineered):

| Model               | Description                         | Artifact       |
|:--------------------|:------------------------------------|:---------------|
| Logistic Regression | Scaled + Logistic Regression        | `model1.pkl`   |
| Decision Tree       | Scaled + Decision Tree              | `model2.pkl`   |
| Random Forest       | Scaled + Tuned Random Forest        | `model3.pkl`   |

- **Split:** 80/20 stratified train/test  
- **Tuning:** `GridSearchCV` on Random Forest parameters  
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

```bash
# Evaluate locally
accuracy_score, classification_report, confusion_matrix
```  

---

## 🌐 Flask Application & API
Go to **`app/main.py`**:

```yaml
GET  /           → Renders responsive web form
POST /predict    → HTML result with inputs & prediction
POST /<key>/evaluate → JSON API (keys: lr, dt, rf)
```

**Request JSON**:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
**Response JSON**:
```json
{ "prediction": 0, "species": "setosa" }
```

---

## 🖥️ Web Interface
- **Bootstrap 4** for a gradient background, rounded cards, and accessible forms.  
- **Templates**:
  - `templates/index.html` (input form)  
  - `templates/result.html` (prediction display)

---

## 📬 API Testing with Postman
12 requests (4 inputs × 3 models). All returned correct species with <20 ms latency:

```bash
POST http://127.0.0.1:5000/rf/evaluate
# JSON body: {"sepal_length":5.9,...}
```  

---

## 🗂️ Project Structure
```text
project_root/
├── app/
│   ├── main.py
│   ├── model1.pkl
│   ├── model2.pkl
│   ├── model3.pkl
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/style.css
├── notebooks/model_training.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔧 Installation & Usage
1. **Clone** repo & `cd project_root`  
2. **Create venv**: `python -m venv venv`  
3. **Activate**: `. venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)  
4. **Install**: `pip install -r requirements.txt`  
5. **Run**: `python app/main.py`  
6. **Browse UI**: `http://127.0.0.1:5000/`  
7. **Test API**: Postman or `curl` on `/lr/evaluate`, `/dt/evaluate`, `/rf/evaluate`

---

## 👥 Group Members & Roles
| Member   | Role                                            |
|:---------|:------------------------------------------------|
| Harjot   | Data exploration, UI design, Postman testing    |
| Vaidehi  | Model training, tuning, Flask API development   |

---

## 📚 References
- [scikit-learn](https://scikit-learn.org/stable/)  
- [Flask](https://flask.palletsprojects.com/)  
- [Bootstrap 4](https://getbootstrap.com/docs/4.6/)  
- [Pandas](https://pandas.pydata.org/docs/)  
- [Seaborn](https://seaborn.pydata.org/)

---

## 🔮 Future Work
- Dockerize the Flask service  
- Add CI/CD pipelines for automated testing  
- Expand to larger flower datasets  
- Implement authentication & logging for production

---

© 2025 Iris Project Team

