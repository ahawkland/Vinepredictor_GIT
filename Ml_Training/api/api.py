from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel
import Ml_Training.src as ml

app = FastAPI()


class DataLine(BaseModel):
    name: str
    features: list


class UpdateDataLine(BaseModel):
    name: Optional[str] = None
    features: Optional[list] = None


vine1 = DataLine(name="Bull blood",
                 features=[5, 14.2, 1.76, 2.45, 15.2, 112, 3.27, 3.39, 0.34, 1.97, 6.75, 1.05, 2.85, 1450])

entities = {
    1: vine1
}


@app.post("/create-entity/{entity_id}", )  # .post - to create something
def create_entity(entity_id: int, entity: DataLine):
    if entity_id in entities:
        return {"Error": "Entity already exists"}
    entities[entity_id] = entity
    return entities[entity_id]


@app.put("/update-entity/{entity_id}")  # .put - to update
def update_entity(entity_id: int, entity: UpdateDataLine):
    if entity_id not in entities:
        return {"Error": "Entity does not exist"}  # this checks whether the id already exists
    if entity.name is not None:
        entities[entity_id].name = entity.name
    if entity.features is not None:
        entities[entity_id].features = entity.features
    return entities[entity_id]


@app.delete("/delete-entity/{entity_id}")
def delete_entity(entity_id: int):
    if entity_id not in entities:
        return {'Error': "Entity does not exists"}

    del entities[entity_id]
    return {"Message": "Entity deleted successfully"}


@app.get("/")  # our own page endpoint
def index():
    return {"Message": "This is a Machine Learning predictor project"}  # python dictionary


@app.get(
    "/get-entity/{entity_id}")  # endpoint parameter is used to return a data relating to an input in the part of the endpoint
def get_entity(entity_id: int = Path(None, description="The ID of the Entity you want to view", gt=0, lt=10)):
    return entities[entity_id]


@app.get("/get-by-name/{entity_id}")  # student id is a path parameter, since it is in the path
def get_entity(*, entity_id: int, name: Optional[str] = None,
               test: int):  # name is a query parameter: its only in the function
    for entity_id in entities:
        if entities[entity_id]['name'] == name:
            return entities[entity_id]
    return {"Data": "Not found"}


@app.get("/predict/{entity_id}")
def predict(entity_id: int):
    if entity_id in entities:
        prediction = ml.predictor(entities[entity_id].features)
        return {'For this vine:' : entities[entity_id].name,
                'The predicted quality is:': int(prediction[0])}
    return {"Data": "Not found"}


def main():
    print('Type prediction of vine no 1:\n', predict(1))


if __name__ == '__main__':
    main()
