import pandas as pd
import src.vinepredictor.pathconfig as pc
from src.vinepredictor.model.train import execute_all
from src.vinepredictor.model.train import read_configs


def generate_model(model_list):
    path_ = pc.PathConfig()
    data_path = path_.data.joinpath('vine_dataset.csv')
    df = pd.read_csv(data_path)
    df_x = df.drop('Type', axis=1)
    df_y = df[['Type']]
    metrics = execute_all(model_list, df_x, df_y, filename='vine_model',
                          config=read_configs(path_.model_config), savemodel=True)


def main():
    model_list = ['random_forest']
    generate_model(model_list)


if __name__ == '__main__':
    main()
