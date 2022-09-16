####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np

from utils import text_clean

####################################################################################################################################
####################################################################################################################################
#############################################################CLASSES################################################################

#########################################################################################################
# Class that returns the probability estimate of a binary event from an ensemble of models:

class Ensemble:
    """
    Class that returns the probability estimate of a binary event from an ensemble of models.
    
    The "predict_proba" method supports objects that have either a "predict" or a "predict_proba" method for
    calculating the proability of the positive class occurrence. In the later case, it is supposed that the
    probability estimates of both classes are returned by the original "predict_proba" method, so the estimate
    for the positive class is extracted by indexing the output of the original method.
    
    Note that the argument "inputs" of the "predict_proba" method should have values of variables sorted in the
    same way as the training data.
    
    Arguments for initialization:
        :param models: collection of models for the ensemble.
        :type models: list.
    
    Methods:
        "predict_proba": takes inputs as arguments and returns probability predictions.
    """
    def __init__(self, models):
        self.models = models
        self.model_types = [str(type(model)).split('<class ')[1].split('.')[0].replace("'", "") for
                            model in models]
    
    def predict_proba(self, inputs, statistic='mean'):
        """
        Method for predicting probabilities.
        
        :param inputs: matrix whose rows are instances and columns are features. These features should
        be ordered in the same way as the dataset used for training.
        :type inputs: numpy ndarray or pandas dataframe.
        :param statistic: indicates how the ensemble predictions should be calculated. Choose between
        "mean" and "max", for mean and maximum values, respectively.
        :type statistic: string.
        
        :return: probability predictions for the instances present in "inputs" data.
        :rtype: list.
        """
        preds = []
        
        # Loop over models:
        for model in self.models:
            # Checking which method should be used for predicting probabilities:
            if 'predict_proba' in dir(model):
                preds.append([p[1] for p in model.predict_proba(inputs)])
            
            else:
                preds.append([p for p in model.predict(inputs)])
        
        if statistic=='mean':
            return [sum(i)/len(i) for i in zip(*preds)]
        
        elif statistic=='max':
            return [max(i) for i in zip(*preds)]

#########################################################################################################
# Class that transforms a raw vector of inputs into a transformed one for feeding the model:

class Inputs:
    """
    Class that transforms a raw vector of inputs into a transformed one for feeding the model.
    
    Class that transforms a raw vector of inputs into a transformed one. The transformations are the
    following: log-transformation and standard scaling of continuous variables, missing values treatment
    (imputation of 0 and creation of dummies indicating the occurrence of missings), and one-hot encoding
    according to what the model is expecting.
    The main outcome from using this class is the attribute "inputs" that follows the calling of the
    method "transform". It contains data consistent with the variables used to train the model, while the
    order of variables in this vector is precisely the same of those used for training the model.
    
    Arguments for initialization:
        :param raw_inputs: variables that should be transformed to feed the model. Supposes that feature
        engineering had already been done over attributes of orders.
        :type raw_inputs: dictionary (or pandas series).
        :param schema: names of inputs as keys and their data types as values. This dictionary should be
        ordered in the exact same way as the matrix used for training the model.
        :type schema: dictionary.
        :param stats: means and standard deviations of continuous inputs for standard scaling. It has two
        keys: "means" and "stds". For each one of them, the same inputs names appear as keys, while their
        values are the corresponding statistics.
        :type stats: dictionary.
        :param numerical_transf: indicates whether numerical transformations (log-transform and standard scaling)
        should be applied over the incoming instance.
        :type numerical_transf: boolean.
    
    Methods:
        "transform": should be called in order to transform the raw inputs into a vector (pandas series)
        of inputs to feed the model and get a probability estimate in return.
    """
    def __init__(self, raw_inputs, schema, stats, numerical_transf=True):
        self.raw_inputs = pd.Series(raw_inputs)
        self.schema = schema
        self.stats = stats
        self.numerical_transf = numerical_transf
        self.variables = list(schema.keys())
    
    def transform(self):
        # Creates the attribute of transformed inputs:
        self.inputs = self.raw_inputs.copy()
        
        # Defines continuous variables for transformation and iterate over them:
        if self.numerical_transf:
            to_transform = [v for v in self.schema if v in self.stats['means']]
            for v in to_transform:
                raw_value = self.raw_inputs[v.replace('L#', '')]
                raw_value = 0 if raw_value < 0 else raw_value
                self.inputs[v] = (np.log(raw_value+0.0001)-self.stats['means'][v])/self.stats['stds'][v]

        # Defines variables with missings during training and iterate over them creating dummies:
        has_missings = [v.split('NA#')[1] for v in self.schema.keys() if 'NA#' in v]
        for v in has_missings:
            self.inputs['NA#'+v] = 1 if pd.isna(self.raw_inputs[v]) else 0

        # Imputing missings:
        self.inputs.fillna(0, inplace=True)

        # Transformation of the vector into a dataframe:
        inputs_df = pd.DataFrame(self.inputs).T

        # Defines categorical variables for transformation and iterates over them:
        to_ohe = sorted(list(set([v.split('#')[1] for v in self.schema.keys() if 'C#' in v])))
        for f in to_ohe:
            inputs_df[f] = inputs_df[f].apply(text_clean)
            dummies = pd.get_dummies(inputs_df[f])
            dummies.columns = [f'C#{f}#{c.upper()}' for c in dummies.columns]
            inputs_df = pd.concat([inputs_df, dummies], axis=1, sort=False)

        # Creating remaining dummy variables:
        for v in [v for v in self.schema.keys() if 'C#' in v]:
            if v not in inputs_df.columns:
                inputs_df[v] = 0
        
        not_found = [v for v in self.variables if v not in list(inputs_df.columns)]
        if len(not_found) > 0:
            raise ValueError(f'As seguintes variáveis não foram encontradas na instância transformada: {not_found}.')
        
        # Converts the (transformed) input vector into a pandas series:
        self.inputs = inputs_df[self.variables].iloc[0]

#########################################################################################################
# Class that transforms a raw vector of inputs into a transformed one, and then feeds the model:

class Model:
    """
    Class that transforms a raw vector of inputs into a transformed one, and then feeds the model.
    
    The incoming instance should be either a pandas series or a dictionary. Besides, it should contain the same
    raw variables as those expected to feed the model after transformations. Such transformations are applied
    by the initialization of the Inputs class, followed by the calling of "transform" method. The final prediction,
    in its turn, is based on an instance of the Ensemble class, which is inserted as an argument during the
    initialization.
 
    Arguments for initialization:
        :param raw_variables: collection of raw variables that should be transformed prior to model
        prediction.
        :type raw_variables: list.
        :param ensemble: ensemble of models for prediction.
        :type ensemble: Ensemble object.
        :param schema: names of inputs as keys and their data types as values. This dictionary should be
        ordered in the exact same way as the matrix used for training the model.
        :type schema: dictionary.
        :param stats: means and standard deviations of continuous inputs for standard scaling. It has two
        keys: "means" and "stds". For each one of them, the same inputs names appear as keys, while their
        values are the corresponding statistics.
        :type stats: dictionary.
        :param numerical_transf: indicates whether numerical transformations (log-transform and standard scaling)
        should be applied over the incoming instance.
        :type numerical_transf: boolean.
    
    Methods:
        "prediction": transforms the raw instance and produces a probability estimate of fraud.
    """
    def __init__(self, raw_variables, ensemble, schema, stats, numerical_transf=True):
        self.raw_variables = raw_variables
        self.ensemble = ensemble
        self.schema = schema
        self.stats = stats
        self.numerical_transf = numerical_transf
    
    def predict(self, raw_instance, dictionary=False):
        """
        Method that receives the raw instance, transform it accordingly and returns the probability
        estimate of fraud.
        
        :param raw_instance: vector of variables without any transformation. It will be transformed and,
        then, feed the model to produce the score.
        :type raw_instance: dictionary (or pandas series).
        
        :return: the probability estimate of fraud.
        :rtype: float.
        """
        if dictionary:
            # Checking if any necessary raw variable is unavailable for transformation:
            available_variables = list(raw_instance.index) if isinstance(raw_instance, pd.Series) else list(raw_instance.keys())
            not_found = [v for v in self.raw_variables if v not in available_variables]
            if len(not_found) > 0:
                raise KeyError(f'As seguintes variáveis brutas não foram encontradas na instância enviada: {not_found}.')
                
            # Transforming the raw instance into a transformed vector of input variables:
            self.instance = Inputs(raw_inputs=raw_instance, schema=self.schema, stats=self.stats,
                                   numerical_transf=self.numerical_transf)
            self.instance.transform()

            return self.ensemble.predict_proba(self.instance.inputs.values.reshape(1,-1))[0]

        else:
            if raw_instance.shape[1] != len(self.raw_variables):
                raise ValueError(f'Há uma diferença de {raw_instance.shape[1] - len(self.raw_variables)} variáveis entre a instância e os dados de treino!')
            
            predictions = []
            
            # Loop over instances:
            for raw_inst in raw_instance:
                raw_instance_dict = dict(zip(self.raw_variables, raw_inst))

                # Transforming the raw instance into a transformed vector of input variables:
                self.instance = Inputs(raw_inputs=raw_instance_dict, schema=self.schema, stats=self.stats,
                                       numerical_transf=self.numerical_transf)
                self.instance.transform()
                
                # Predictions:
                predictions.append(self.ensemble.predict_proba(self.instance.inputs.values.reshape(1,-1))[0])
            
            return np.array(predictions)
