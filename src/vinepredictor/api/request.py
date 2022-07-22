import requests


def get_prediction(request_json: dict) -> dict:
    req_features = request_json['feature']
    resp = requests.post("http://127.0.0.1:8000/predict", json=req_features)
    #print(resp)
    if resp.status_code == 200:
        json_resp = resp.json()  # {"prediction": 1}
        return {"prediction": json_resp}
    else:
        return {"prediction": None}


def post_precess(text: str) -> dict:
    into_list = text.split(', ')
    into_dict = {"feature": into_list}
    return into_dict


def prediction_request(request_from_user: str) -> dict:
    request = post_precess(request_from_user)
    prediction = get_prediction(request)
    if prediction["prediction"] is None:
        return {"Error": "Could not predict, the input is malformed"}
    else:
        return prediction


def main():
    request_from_user: str = "0, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065"
    #request_from_user: str = "xyz"
    print(prediction_request(request_from_user))


if __name__ == "__main__":
    main()
