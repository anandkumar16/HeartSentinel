## HeartSentinel

**HeartSentinel** is an end-to-end machine learning web application capable of predicting the likelihood of heart failure in patients. Built with **Python** and **Streamlit**, it utilizes a **K-Nearest Neighbors (KNN)** classifier trained on a clinical dataset of 918 patient records.

---

## Project Overview
Cardiovascular diseases (CVDs) are the number one cause of death globally. This tool aims to assist in the early detection of heart abnormalities by analyzing 11 key physiological indicators.

### Key Features
* **Real-time Prediction:** Instant classification of "High Risk" or "Low Risk" based on user input.
* **Clinical Dataset:** Trained on the **Heart Failure Prediction Dataset** (918 samples) combining data from 5 distinct heart disease databases.
* **Robust Preprocessing:** Implements **One-Hot Encoding** for categorical variables and **StandardScaler** to normalize physiological data ranges.
* **Interactive UI:** A user-friendly dashboard built with Streamlit for easy data entry.

---

##  Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (K-Neighbors Classifier)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

---

## Project Structure
```
Heart_Stroke_Project/
├── app.py                 
├── train_model.py         
├── heart.csv              
├── requirements.txt       
├── README.md             
├── knn_heart_model.pkl    
├── heart_scaler.pkl       
└── heart_columns.pkl      
```

## Installation & Local Setup
Follow these steps to run the project on your local machine:

1. Clone the Repository

```
git clone https://github.com/anandkumar16/HeartSentinel.git
```
2. Install Dependencies

```
pip install -r requirements.txt
```
3. Train the Model

```
python train_model.py
```
4. Run the Application

```
streamlit run app.py
```
Model Details
Algorithm: K-Nearest Neighbors (KNN).

Why KNN? It is effective for classification tasks where relationships between physiological markers (like Age, Cholesterol, MaxHR) are non-linear.

Preprocessing: *Categorical features (Sex, ChestPainType, etc.) are converted to numeric via One-Hot Encoding*.

Numerical features are scaled to a standard range to prevent larger values (like Cholesterol) from dominating the distance calculation.


