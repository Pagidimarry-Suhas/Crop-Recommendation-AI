# 🌾 AI-Based Crop Recommendation System

An end-to-end machine learning project that recommends the most suitable crop
for a given field based on soil nutrient levels and weather conditions.

---

## 📁 Project Structure

```
crop_recommendation/
├── app.py                          ← Streamlit web application
├── requirements.txt
├── data/
│   ├── generate_dataset.py         ← Synthetic dataset generator
│   └── crop_data.csv               ← Generated dataset (2 200 rows, 22 crops)
├── models/
│   ├── best_model.pkl              ← Best trained model (Random Forest)
│   ├── scaler.pkl                  ← StandardScaler
│   ├── label_encoder.pkl           ← LabelEncoder
│   └── metadata.pkl                ← All model results & feature info
├── notebooks/
│   └── crop_eda_and_modelling.ipynb
├── reports/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── eda_scatter.png
└── src/
    ├── train_models.py             ← Full training pipeline
    └── predictor.py                ← Reusable prediction module
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset 
```bash
python data/generate_dataset.py
```

### 3. Train models
```bash
python src/train_models.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```

### 5. Use the predictor in code
```python
from src.predictor import CropPredictor

p = CropPredictor()
result = p.predict(N=80, P=45, K=40, temperature=23,
                   humidity=82, ph=6.5, rainfall=200)
print(result["best_crop"])     # → rice
print(result["confidence"])    # → 0.584
```

---

## 🌿 Features

| Feature      | Description                          | Unit    |
|-------------|--------------------------------------|---------|
| N           | Nitrogen content                     | kg/ha   |
| P           | Phosphorus content                   | kg/ha   |
| K           | Potassium content                    | kg/ha   |
| temperature | Average ambient temperature          | °C      |
| humidity    | Relative humidity                    | %       |
| ph          | Soil pH                              | —       |
| rainfall    | Annual rainfall                      | mm      |

---

## 🤖 Models & Results

| Model          | CV Accuracy | Test Accuracy |
|---------------|-------------|---------------|
| Random Forest | 95.97%      | **96.59%** ★  |
| SVM (RBF)     | 96.25%      | 96.36%        |
| KNN           | 95.57%      | 95.00%        |

All models were tuned using `GridSearchCV` with 5-fold cross-validation.

---

## 🌾 Supported Crops (22 classes)

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean,
Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon,
Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 📊 App Tabs

| Tab                | Contents                                              |
|--------------------|-------------------------------------------------------|
| 🔮 Prediction      | Real-time crop recommendation with confidence chart   |
| 📊 EDA & Insights  | Distribution plots, correlation heatmap, box plots    |
| 🤖 Model Performance | Accuracy comparison, confusion matrix, importances  |
| 📋 Dataset         | Browse raw data & descriptive statistics             |
