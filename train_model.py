import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    print("Loading data...")
    # 1. Load the data
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        print("Error: heart.csv not found! Please create it first.")
        return

    # 2. Separate Target and Features
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # 3. One-Hot Encoding (Converting letters to numbers)
    # We must match the columns expected by the app
    X = pd.get_dummies(X, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # 4. Scale the features (Important for KNN!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train the Model
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_scaled, y)

    # 6. Save the files
    print("Saving model artifacts...")
    joblib.dump(knn, "knn_heart_model.pkl")
    joblib.dump(scaler, "heart_scaler.pkl")
    joblib.dump(X.columns, "heart_columns.pkl")
    
    print("Done! .pkl files created successfully.")

if __name__ == "__main__":
    train_and_save_model()