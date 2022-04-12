import typing as t
import pickle


def save_model(model: t.Any, filename: t.Any, modelname: t.Any) -> str:
    """
    Save the specified model with pickle under the given filename + the modelname.
    Also prints the saved file name.
    """
    saved_model_name = f"{filename}_{modelname}"
    filepath = ""
    with open(f"{filepath}{saved_model_name}", 'wb') as file_:
        pickle.dump(model, file_)
    print('Model Saved: ', saved_model_name)
    return saved_model_name
