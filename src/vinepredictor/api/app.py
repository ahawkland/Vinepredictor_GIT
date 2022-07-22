from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from src.vinepredictor.pathconfig import PathConfig
from src.vinepredictor.model.predictor import Predictor
from src.vinepredictor.api.request import prediction_request


app = FastAPI()
templates = Jinja2Templates(directory="src/vinepredictor/api/templates")


model_path = PathConfig().output.joinpath('vine_model_ran.pickle')
predictor = Predictor(model_path)
predictor.init()


@app.get("/")  # our own page endpoint
def home(request: Request):
    """
    Displays the vinepredictor indexpage
    :return:
    """
    return templates.TemplateResponse("home.html", {
        "request": request
    })



@app.post("/predict")
def predict_app(features: list):
    prediction = predictor.predict(features)
    return {'The predicted vine class is :': int(prediction[0])}


@app.post("/prediction_request")
async def prediction_request_app(features_from_user):
    print("prediction_request_app started")
    return prediction_request(features_from_user)


def main():
    print('Type prediction of vine no 1:\n', predict_app([0, 14.23, 1.71, 2.43, 15.6, 127, 2.8,
                                                          3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]))


if __name__ == '__main__':
    main()
