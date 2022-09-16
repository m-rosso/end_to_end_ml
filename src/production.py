####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
from typing import Union, Optional
from pydantic import BaseModel

import lightgbm as lgb
import xgboost as xgb

from transformations import Pipeline
from feat_eng import known_related_apps, related_malwares

####################################################################################################################################
####################################################################################################################################
#####################################################FUNCTIONS AND CLASSES##########################################################

####################################################################################################################################
# Class that returns predictions from an ensemble of trained models:

class Ensemble:
    """
    Class that returns predictions from an ensemble of trained models.

    Arguments for initialization:
        :param models: collection of trained models to create an ensemble.
        :type models: list or tuple.
        :param statistic: statistic for producing ensemble predictions from individual predictions.
        :type statistic: string.
        :param weights: weights for individual predictions. Used when "statistic" is set to "weighted_mean".
        Choose among "weighted_mean", "mean" and "median".
        :type weights: list or tuple.
        :param task: predictive task. Choose among "binary_class", "multi_class" and "regression".
        :type task: string.
    
    Methods:
      "predict": returns the ensemble prediction for each input in the provided batch. It allows label prediction
      when "predict_class" is set to True and uses predicted scores together with a provided value of threshold for
      binary classification tasks.
      "weighted_mean": static method that returns a weighted mean from weights and values.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, models: Union[list, tuple], statistic: str = 'weighted_mean',
                 weights: Optional[Union[list, tuple]] = None, task: str = 'binary_class'):
        self.models = models
        self.statistic = statistic
        self.weights = weights
        self.task = task
    
    def predict(self, inputs: pd.DataFrame,
                predict_class: Optional[bool] = False, threshold: Optional[float] = 0.5) -> list:
        """
        Method that returns the ensemble prediction for each input in the provided batch.

        :param inputs: batch of inputs for producing ensemble predictions.
        :type inputs: dataframe.
        :param predict_class: for binary or multiclass classification tasks, it indicates whether
        predicted labels or predicted scores should be returned.
        :type predict_class: boolean.
        :param threshold: for binary classification tasks, scores higher than this value lead to
        prediction of positive class.
        :type threshold: float.

        :return: ensemble prediction for each input in the provided batch.
        :rtype: list.
        """
        if predict_class:
            return self.__predict_class(inputs, threshold)
        else:
            return self.__ensemble_values(inputs)

    def __predict_class(self, inputs: pd.DataFrame, threshold: Optional[float] = 0.5) -> list:
        predictions = self.__ensemble_values(inputs)

        if self.task == 'binary_class':
            return [1 if p > threshold else 0 for p in predictions]

        elif self.task == 'multi_class':
            return [np.argmax(p) for p in predictions]

        else:
            raise ValueError('For classes prediction, please choose "task" as "binary_class" or "multi_class".')

    def __ensemble_values(self, inputs: pd.DataFrame) -> list:
        predictions = self.__predict_values(inputs)

        if self.statistic == 'weighted_mean':
            return [self.weighted_mean(p, self.weights) for p in zip(*predictions)]
        
        elif self.statistic == 'mean':
            return [np.nanmean(p) for p in zip(*predictions)]
        
        elif self.statistic == 'median':
            return [np.nanmedian(p) for p in zip(*predictions)]
        
        else:
            raise ValueError('Please, choose "statistic" as "weighted_mean", "mean", or "median".')

    def __predict_values(self, inputs: pd.DataFrame) -> np.ndarray:
        if self.task == 'binary_class':
            return self.__binary_predict(inputs)

        elif self.task == 'multi_class':
            return self.__multi_predict(inputs)
        
        elif self.task == 'regression':
            return self.__reg_predict(inputs)

        else:
            raise ValueError('Please, choose "task" as "binary_class", "multi_class", or "regression".')
    
    def __binary_predict(self, inputs: pd.DataFrame) -> np.ndarray:
        predictions = []

        # Loop over models:
        for model in self.models:
            if 'predict_proba' in dir(model):
                predictions.append(model.predict_proba(inputs)[:, 0])

            elif isinstance(model, lgb.Booster):
                predictions.append(model.predict(inputs))

            elif isinstance(model, xgb.Booster):
                xg_inputs = xgb.DMatrix(data=inputs)
                predictions.append(model.predict(xg_inputs))

            else:
                raise ValueError(f'Model {self.models.index(model)} is not from sklearn, LightGBM or XGBoost.')

        return predictions

    def __reg_predict(self, inputs: pd.DataFrame) -> np.ndarray:
        predictions = []

        # Loop over models:
        for model in self.models:
            if 'predict_proba' in dir(model):
                predictions.append(model.predict_proba(inputs))

            elif isinstance(model, lgb.Booster):
                predictions.append(model.predict(inputs))

            elif isinstance(model, xgb.Booster):
                xg_inputs = xgb.DMatrix(data=inputs)
                predictions.append(model.predict(xg_inputs))

            else:
                raise ValueError(f'Model {self.models.index(model)} is not from sklearn, LightGBM or XGBoost.')
        
        return predictions

    def __multi_predict(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.__reg_predict(inputs)

    @staticmethod
    def weighted_mean(values: Union[list, tuple], weights: Union[list, tuple]) -> float:
        """
        Function that returns a weighted mean from weights and values.

        :param values: .
        :type values: list or tuple.
        :param weights: .
        :type weights: list or tuple.

        :returns: weighted mean of the provided collection of values.
        :rtype: float.
        """
        return sum([v*w for v, w in zip(values, weights)])

####################################################################################################################################
# Class for returning predictions from a trained model for raw inputs:

class Model:
    """
    Class for returning predictions from a trained model for raw inputs.

    Arguments for initialization:
        :param schema: variables that are expected to be sent to the model
        without any transformation together with their respective data types.
        :type schema: dictionary.
        :param pipeline: declared object to transform a raw input according
        to the pipeline used during model training.
        :type pipeline: object of Pipeline class.
        :param Ensemble: declared object to produce a prediction for a
        transformed input.
        :type Ensemble: object of Ensemble class.
        :param variables: names of variables that were used for training the
        model and, therefore, that should be used for producing predictions.
        :type variables: list.
    
    Methods:
        "predict": method that produces a prediction for a raw input data point
        given the expected schema, the pipeline of transformations, the ensemble
        of trained models and the list of variables (predictors), all provided
        during the initialization of this class.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, schema: dict, pipeline: Pipeline, ensemble: Ensemble, variables: list):
        self.schema = schema
        self.pipeline = pipeline
        self.ensemble = ensemble
        self.variables = variables

    def predict(self, input_data: Union[dict, np.array], training_data: pd.DataFrame) -> np.array:
        """
        Method that produces a prediction for a raw input data point given the expected schema, the
        pipeline of transformations, the ensemble of trained models and the list of variables
        (predictors), all provided during the initialization of this class.

        :param input_data: raw input data point for which a prediction should be produced.
        :type input_data: dictionary or array.
        :param training_data: training data as expected by the fitted pipeline.
        :type training_data: dataframe.

        :return: prediction for the provided input data point.
        :rtype: array.
        """
        if isinstance(input_data, np.ndarray):
            if len(input_data) != len(list(self.schema.keys())):
                raise ValueError('There is an insufficient number of input variables in this data point vector.')
            input_data = dict(zip(list(self.schema.keys()), input_data))
        
        # Checking the schema:
        self.__check_schema(input_data=input_data)

        if isinstance(input_data, dict):
            input_data = pd.DataFrame(data=input_data, index=[0])
            transf_input = input_data.copy()
        else:
            raise TypeError('"input_data" should be a numpy array or a dictionary.')

        # Cleaning data according to the procedure followed during model training:
        transf_input = self.__cleaning_data(input_data=transf_input)

        # Feature engineering from raw input variables:
        transf_input = self.__feature_engineering(input_data=transf_input, training_data=training_data)

        # Data transformation according to the procedure followed during model training:
        _, transf_input = self.pipeline.transform(data_list=[transf_input], training_data=training_data)
        transf_input = transf_input[0]
        transf_input = transf_input[self.variables]

        # Prediction from the ensemble of trained models:
        return self.ensemble.predict(inputs=transf_input, predict_class=False)
    
    def __cleaning_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Removing signs from the text of related apps:
        input_data['related_apps'] = input_data['related_apps'].apply(
            lambda x: x if pd.isna(x) else x.replace('{', '').replace('}', '')
        )

        return input_data
    
    def __feature_engineering(self, input_data: pd.DataFrame, training_data: pd.DataFrame) -> pd.DataFrame:
        # Creating the variable with the number of related apps:
        input_data['num_related_apps'] = input_data['related_apps'].apply(
            lambda x: np.NaN if pd.isna(x) else len(x.split(', '))
        )

        # Creating the variable that indicates the number of words in a description:
        input_data['num_words_desc'] = input_data.description.apply(
            lambda x: x if pd.isna(x) else len(x.split(' '))
        )

        # Number of known related apps:
        input_data['num_known_apps'] = input_data['related_apps'].apply(
            lambda x: known_related_apps(data=training_data, related_apps=x)
        )

        # Share of related apps that are known:
        input_data['share_known'] = input_data['num_known_apps']/input_data['num_related_apps']

        # Number of known related apps that are malwares:
        input_data['num_known_malwares'] = input_data['related_apps'].apply(
            lambda x: related_malwares(data=training_data, related_apps=x)
        )

        # Share of known related apps that are malwares:
        input_data['share_known_malwares'] = input_data['num_known_malwares']/input_data['num_known_apps']

        return input_data
    
    def __check_schema(self, input_data: dict) -> None:
        missing_vars = [c for c in list(self.schema.keys()) if c not in list(input_data.keys())]
        if len(missing_vars) > 0:
            raise ValueError(f'The following variables were not provided: {missing_vars}.')
        
        strings = [v for v, t in self.schema.items() if t=='str']
        numerics = [v for v, t in self.schema.items() if t!='str']

        strings_wrong = [v for v in strings if (isinstance(input_data[v], str)==False) &
                         (pd.isna(input_data[v])==False)]
        numeric_wrong = [v for v in numerics if isinstance(input_data[v], str)]
        wrong_type = []
        wrong_type.extend(strings_wrong)
        wrong_type.extend(numeric_wrong)

        if len(wrong_type) > 0:
            raise TypeError(f'The following variables have the wrong data type: {wrong_type}.')

        return

####################################################################################################################################
# Class for defining data type of requests:

class UserRequestIn(BaseModel):
    request_id: str
    input_data: dict
