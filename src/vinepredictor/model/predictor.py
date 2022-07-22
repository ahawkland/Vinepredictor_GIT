from src.vinepredictor.pathconfig import PathConfig
import src.vinepredictor.model.modelio as io
from pathlib import Path
import numpy as np


class Predictor:
    """
    This class is to circumvent to load the model every time a prediction is done.
    model_path: where the model to be loaded is located
    """
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None

    def init(self) -> None:
        output_path = self.model_path
        self.model = io.read_model(output_path)

    def predict(self, features: bytearray) -> int:
        features = np.reshape(features, (1, -1))
        prediction = self.model.predict(features)
        return prediction


def main():
    model_path = PathConfig().output.joinpath('vine_model_ran.pickle')
    predictor = Predictor(model_path)
    predictor.init()

    prediction = predictor.predict([0, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065])
    print('prediction:', prediction)


if __name__ == '__main__':
    main()
