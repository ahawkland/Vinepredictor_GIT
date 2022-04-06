from unittest import TestCase
import pandas as pd

from MlProject.main import model_repo
from MlProject.src.train import execute_all


class TestTrain(TestCase):
    def test_execute_all(self):
        df = pd.read_csv('test_df.csv')
        df_x = df[['Flavanoids', 'Dilution', 'Phenols']]
        df_y = df[['Type']]
        execute_all(model_repo, df_x, df_y, 'vine_model', savemodel=False)
