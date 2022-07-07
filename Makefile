python setup.py bdist_wheel
pip install -e .

uvicorn src.vinepredictor.api.app:app  --reload

