import pandas as pd
from src.preprocess import preprocess_data

def test_preprocess_data():
    data = {
        'Age': [22, None],
        'Sex': ['male', 'female'],
        'Embarked': ['S', None],
        'Name': ['John', 'Jane'],
        'Survived': [1, 0]
    }
    df = pd.DataFrame(data)
    processed = preprocess_data(df)
    assert 'Sex_male' in processed.columns
    assert processed.isnull().sum().sum() == 0