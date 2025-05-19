import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.preprocess import load_data, preprocess_data

def train_and_save_model():
    df = load_data("data/titanic.csv")
    df = preprocess_data(df)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    
    joblib.dump(model, "model.pkl")
    return acc

if __name__ == "__main__":
    train_and_save_model()
