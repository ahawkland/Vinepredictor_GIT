from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel
from src.vinepredictor.model.predictor import predictor

app = FastAPI()

@app.get("/")  # our own page endpoint
def index():
    return {"Message": "This is a Machine Learning predictor project for Vine quality"}


@app.post("/predict")
def predict_app(features: list):
    prediction = predictor(features)
    return {'The predicted vine class is :': int(prediction[0])}


def main():
    print('Type prediction of vine no 1:\n', predict_app([0, 14.23, 1.71, 2.43, 15.6, 127, 2.8,
                                                          3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]))


if __name__ == '__main__':
    main()
