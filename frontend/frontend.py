import streamlit as st
import requests
import logging

logging.basicConfig(
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s", level=logging.INFO
)


def get_prediction(request_json: list) -> dict:
    """This function communicates with the backend server prediction app
    request_json: a dictionary containing the list of features
    :returns the prediction in json"""
    req_features = request_json
    resp = requests.post("http://127.0.0.1:8000/predict", json=req_features)
    if resp.status_code == 200:
        json_resp = resp.json()  # {"prediction": 1}
        return {"prediction": json_resp}
    else:
        return {"prediction": None}


def post_process(text) -> list:
    """This function process the user input string into a list of floats
    :returns a list of floats"""
    int_list = text.split(", ")
    into_list = [float(i) for i in int_list]
    return into_list


# --- Setting up the page ---
st.set_page_config(page_title="Vinepredictor app", layout="wide")

# --- Header Section ---
st.header("Vinepredictor App")
st.text("please enter your features of your vine")
input_features = st.text_area(label="Features")

# --- User input and prediction ---
if input_features:  # after user enters the features
    st.text("Your features:")
    st.text(input_features)
    logging.info(f"input_features_type: {input_features}")
    logging.info(f"input_features: {input_features}")
    try:  # if the input is processable the prediction is run
        processed_features = post_process(input_features)
        response = get_prediction(processed_features)
        st.text(
            f"The prediction of your input features is: {response['prediction']['prediction']}"
        )
    except Exception as e:  # if the input is not adequate error returns
        logging.info(e)
        st.text("There is something wrong with your input, please check.")
        st.text(str(e))
