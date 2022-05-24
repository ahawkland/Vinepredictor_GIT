from unittest import TestCase
import pandas as pd
import os
import Ml_Training.src as ml
from sklearn.datasets import load_iris

class TestTrain(TestCase):
    def test_execute_all_vine(self):
        path_ = ml.PathConfig()
        model_list = ['random_forest']
        df = pd.read_csv('test_df.csv')
        df_x = df[['Flavanoids', 'Dilution', 'Phenols']]
        df_y = df[['Type']]
        metrics = ml.execute_all(model_list, df_x, df_y, filename='vine_model',
                                 config=ml.read_configs(path_.model_config), savemodel=False)

    def test_execute_all_iris(self):
        model_list = ['random_forest']
        path_ = ml.PathConfig()
        # Loading data
        irisData = load_iris()
        # Create feature and target arrays
        df_x = pd.DataFrame(irisData.data)
        df_y = pd.DataFrame(irisData.target)
        metrics = ml.execute_all(model_list, df_x, df_y, filename='iris_model',
                                 config=ml.read_configs(path_.model_config), savemodel=True)
        os.remove(path_.output.joinpath('iris_model_ran.pickle'))
        print(10 * '-')
        print('The saved model was deleted as per the training code, not to leave unnecessary files behind')


