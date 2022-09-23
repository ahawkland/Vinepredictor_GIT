import streamlit as st
import sys
sys.path.insert(0, '../src')
#from sys.path('src') import prediction_request

st.set_page_config(page_title="Vinepredictor app", layout="wide")

# --- Header Section ---
st.header("Vinepredictor App")
st.text("please enter your features of your vine")
input_features = st.text_area(label='Features')
st.text("Your features:")

"""
if input_features:
    try:
        response = prediction_request(input_features)
        st.text(f"The prediction of your input features {response}")
    except Exception as e:
        print(e)

"""
def main():
    #print(prediction_request('0, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065'))
    print(hello())


if __name__ == '__main__':
    main()