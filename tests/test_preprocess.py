import pandas as pd
from src.preprocess import preprocess_data

def test_handle_missing_values():
    df = pd.DataFrame({
        "Age": [22, None],
        "Embarked": ["S", None],
        "Survived": [1, 0],
        "Sex": ["male", "female"],
        "Pclass": [1, 3]
    })
    df_processed = preprocess_data(df)
    assert df_processed.isnull().sum().sum() == 0
