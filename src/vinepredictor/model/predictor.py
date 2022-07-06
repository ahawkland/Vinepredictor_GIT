from src.vinepredictor.pathconfig import PathConfig
import src.vinepredictor.model.modelio as io
import numpy as np


def predictor(features: bytearray):
    output_path = PathConfig().output.joinpath('vine_model_ran.pickle')
    loaded_model = io.read_model(output_path)
    features = np.reshape(features, (1, -1))
    prediction = loaded_model.predict(features)
    return prediction


def main():
    prediction = predictor([0, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065])
    print(prediction)


if __name__ == '__main__':
    main()
