from src.train import train_and_save_model

def test_model_accuracy():
    acc = train_and_save_model()
    assert acc >= 0.80
