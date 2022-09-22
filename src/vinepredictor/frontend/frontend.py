import streamlit as st
from src.vinepredictor.api.request import prediction_request

st.header("Vinepredictor App")
st.text("please enter your features of your vine")
input_features = st.text_area(label='Features')
st.text("Your features:", input_features)

if input_features:
    try:
        response = prediction_request(input_features)
        st.text(f"The prediction of your input features {response}")
    except Exception as e:
        print(e)


