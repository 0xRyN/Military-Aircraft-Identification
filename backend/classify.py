import keras
import numpy as np


class Classifier:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def classify(self, image) -> tuple[int, float]:
        list = self.model.predict(image)
        idx = int(np.argmax(list))
        likelihood = list[0][idx]
        return idx, likelihood
