import autokeras as ak
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

cls = ak.ImageClassifier()
cls.fit(x_train, y_train)
