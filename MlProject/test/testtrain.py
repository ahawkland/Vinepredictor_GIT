from unittest import TestCase
import pandas as pd
import os

from MlProject.main import model_repo
from MlProject.src.train import execute_all


class TestTrain(TestCase):
    def test_execute_all(self):
        df = pd.read_csv('test_df.csv')
        df_x = df[['Flavanoids', 'Dilution', 'Phenols']]
        df_y = df[['Type']]
        metrics, best_saved_model = execute_all(model_repo, df_x, df_y, 'vine_model', savemodel=True)
        os.remove(best_saved_model)
        print(10 * '-')
        print('The saved model was deleted as per the training code, not to leave unnecessary files behind')

