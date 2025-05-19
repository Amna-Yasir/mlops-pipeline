import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path='data/titanic.csv'):
    return pd.read_csv(path)
#preprocessing
def preprocess_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df.dropna(inplace=True)
    return df


def normalize_data(df):
    scaler = StandardScaler()
    features = df.drop('Survived', axis=1)
    df_scaled = scaler.fit_transform(features)
    return df_scaled, df['Survived']