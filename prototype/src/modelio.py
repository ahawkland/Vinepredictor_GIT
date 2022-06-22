import typing as t
import pickle
from pathlib import Path
from .pathconfig import PathConfig

PATHFINDER = PathConfig()
CENTRAL_MODEL_REPOSITORY = PATHFINDER.output


def save_model(model: t.Any, filename: t.Any, modelname: t.Any) -> None:
    """
    Save the specified model with pickle under the given filename + the modelname.
    Also prints the saved file name.
    :param model: the trained model
    :param filename: the desired filename which under the model is saved
    :param modelname: the name of the model
    :return: None
    """
    file_name = f"{filename}_{modelname}.pickle"
    with open(CENTRAL_MODEL_REPOSITORY.joinpath(file_name), 'wb') as file_:
        pickle.dump(model, file_)
    print(f'Model Saved as: {file_name}')
    return file_name


def read_model(filename: str) -> t.Any:
    """
    The function loads the specified saved model (filename)
    :param filename: the saved models name
    :return: the model loaded in
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#