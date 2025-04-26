# airlinepricepredictorkaggle
https://colab.research.google.com/drive/1xxl2k10M1OBOfyQ_vHOO_p8iXFn8zTR-

# ✈️ Airline Price Predictor using Random Forest Regression  

This project predicts airline ticket prices using a Random Forest Regression model. It leverages a cleaned dataset containing flight information, and emphasizes **feature engineering using One-Hot Encoding** to handle categorical variables such as airline, city, and flight times.

---

## 📂 Project Files

- `pricepredictorairlinekaggle.ipynb`: Jupyter Notebook containing the full workflow from data exploration, preprocessing, model training, evaluation, and visualization.
- `Clean_Dataset.csv`: Preprocessed flight dataset containing cleaned and engineered features.

---

## 🎯 Objectives

- Explore airline ticket data (cities, airlines, timings, flight class, etc.)
- Preprocess categorical data using **One-Hot Encoding**
- Train a robust regression model to predict flight prices
- Evaluate the model using R², MAE, RMSE
- Visualize the model’s performance (actual vs predicted)

---

## 📊 Dataset Overview

- **Source**: Cleaned Kaggle Dataset (uploaded via Google Drive)
- **Features**:
  - Categorical: Airline, Source City, Destination City, Class, Arrival Time, Departure Time, Number of Stops
  - Numerical: Days Left, Duration
  - Target: `price`

---

## 🔧 Preprocessing Highlights

- Removed unused columns (`Unnamed: 0`, `flight`)
- **Binary Encoding** for class:  
  `Economy = 0`, `Business = 1`
- **Factorization** for `stops`
- ✅ **Extensive One-Hot Encoding** applied to:
  - `airline` → `airline_Indigo`, `airline_AirAsia`, etc.
  - `source_city` → `source_Delhi`, `source_Mumbai`, etc.
  - `destination_city` → `dest_Bangalore`, `dest_Kolkata`, etc.
  - `arrival_time`, `departure_time` → Encoded for various time buckets (e.g., Morning, Night, Late_Night)

> One-Hot Encoding played a crucial role in preparing the dataset for machine learning by ensuring that categorical variables were converted into a format suitable for regression models.

---

## 🤖 Model Training

- **Model**: `RandomForestRegressor` from `scikit-learn`
- **Train-Test Split**: 80/20
- **Training Time**: ~Few seconds with parallel processing (`n_jobs=-1`)

```python
reg = RandomForestRegressor(n_jobs=-1)
reg.fit(X_train, y_train)
