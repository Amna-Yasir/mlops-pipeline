import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from src.train import train_and_save_model

def test_model_accuracy():
    acc = train_and_save_model()
    assert acc >= 0.80
