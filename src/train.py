import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_data, normalize_data
#from .preprocess import load_data, preprocess_data, normalize_data

def train_and_save_model():
    df = load_data()
    df = preprocess_data(df)
    X, y = normalize_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    joblib.dump(model, 'model.pkl')
    return accuracy

if __name__ == '__main__':
    train_and_save_model()