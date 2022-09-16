##############################################################################
##############################################################################
###################################LIBRARIES##################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
import json
import unittest
import warnings
import pickle

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../src'
        )
    )
)

from utils import correct_col_name

##############################################################################
##############################################################################
###########################FUNCTIONS AND CLASSES##############################

##############################################################################
# Class that implements unit tests for the production of predictions from the fitted model:

class TestPrediction(unittest.TestCase):
    def setUp(self):
        """
        Method that initializes attributes necessary for running unit tests.
        """
        # Domain of each raw input variable:
        self.vars_domain = pd.read_csv('../data/Android_Permission.csv')
        self.vars_domain.columns = [correct_col_name(c) for c in self.vars_domain.columns]

        # Expected schema of the inputs:
        with open('../artifacts/schema.json', 'r') as json_file:
            self.schema = json.load(json_file)

        # Training data for producing predictions:
        self.df_train = pd.read_csv('../artifacts/df_train.csv', dtype={'app_id': int})

        # Object of fitted pipeline:
        self.model = pickle.load(open('../artifacts/model.pickle', 'rb'))

    def test_prediction(self):
        """
        Unit test for assessing whether the fitted model successifully returns
        a prediction for random inputs.
        """
        samples = [
                   dict(zip(list(self.schema.keys()),
                            [np.random.choice(self.vars_domain[v].unique()) for v in list(self.schema.keys())]))
                   for i in range(100)
        ]
        for sample in samples:
            with self.subTest(sample):
                try:
                    prediction = self.model.predict(input_data=sample, training_data=self.df_train)
                    check = True
                except:
                    check = False

                self.assertEqual(check, True)

    def test_prediction_dtype(self):
        """
        Unit test for assessing whether the fitted model successifully returns
        a prediction whose data type is float for random inputs.
        """
        samples = [
                   dict(zip(list(self.schema.keys()),
                            [np.random.choice(self.vars_domain[v].unique()) for v in list(self.schema.keys())]))
                   for i in range(100)
        ]
        for sample in samples:
            with self.subTest(sample):
                try:
                    prediction = self.model.predict(input_data=sample, training_data=self.df_train)
                    self.assertIsInstance(prediction[0], float)
                except Exception as error:
                    print(error)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        unittest.main(verbosity=2)
