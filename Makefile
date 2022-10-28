make_package:
    python setup.py bdist_wheel

install_package
    pip install -e .

server:
    run_server:
        uvicorn server.api.app:app  --reload

frontend:
    run_frontend:
        streamlit run frontend.frontend.py
        streamlit run frontend.py

preset_features:
    0, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065

