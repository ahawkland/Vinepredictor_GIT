from fastapi import FastAPI, Request
from src.vinepredictor.pathconfig import PathConfig
from src.vinepredictor.model.predictor import Predictor


app = FastAPI()


model_path = PathConfig().output.joinpath('vine_model_ran.pickle')
predictor = Predictor(model_path)
predictor.init()


@app.get("/")  # our own page endpoint
def home(request: Request):
    """
    Displays the vinepredictor indexpage
    :return: None
    """
    return {'Home:': 'This is a winepredictor application'}


@app.post("/predict")
def predict_app(features: list) -> dict:
    """
    The prediction app on our server
    :param features:
    :return: a dictionary with the prediction as int
    """
    prediction = predictor.predict(features)
    return {'prediction': int(prediction[0])}


def main():
    print('Type prediction of vine:\n', predict_app([0, 14.23, 1.71, 2.43, 15.6, 127, 2.8,
                                                                3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]))

if __name__ == '__main__':
    main()
