import pandas as pd
import Ml_Training.src as ml


def generate_model(model_list):
    path_ = ml.PathConfig()
    data_path = path_.data.joinpath('vine_dataset.csv')
    df = pd.read_csv(data_path)
    df_x = df.drop('Type', axis=1)
    df_y = df[['Type']]
    metrics = ml.execute_all(model_list, df_x, df_y, filename='vine_model',
                             config=ml.read_configs(path_.model_config), savemodel=True)


def main():
    model_list = ['random_forest']
    generate_model(model_list)


if __name__ == '__main__':
    main()
