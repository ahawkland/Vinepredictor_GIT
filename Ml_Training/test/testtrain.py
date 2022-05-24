from unittest import TestCase
import pandas as pd
import os
import Ml_Training.src as ml


class TestTrain(TestCase):
    def test_execute_all(self):
        path_ = ml.PathConfig()
        model_list = ['random_forest']
        df = pd.read_csv('test_df.csv')
        df_x = df[['Flavanoids', 'Dilution', 'Phenols']]
        df_y = df[['Type']]
        metrics = ml.execute_all(model_list, df_x, df_y, filename='vine_model',
                                 config=ml.read_configs(path_.model_config), savemodel=False)


