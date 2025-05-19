import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df
