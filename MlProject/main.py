import pandas as pd
from MlProject.src.model import ModelRepository
from MlProject.src import execute_all

model_registry = ModelRepository()
model_repo = model_registry.get_model_space()
df = pd.DataFrame()


if __name__ == "__main__":
    execute_all(model_repo, df_x, df_y, filename, savemodel=True)
