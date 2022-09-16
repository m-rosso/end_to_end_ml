####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple

from utils import text_clean

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Class that applies the logarithmic transformation over selected numerical variables:

class LogTransformation:
    """
    Class that applies the logarithmic transformation over selected numerical variables.

    Initialization attributes:
        :param to_log: names of columns whose values should be log-transformed.
        :type to_log: list.
        :param eps: minimal value to be added to zero valued observations in order to correctly calculate the logarithm.
        :type eps: float.
        :param drop_raw: indicates whether original variables should be dropped from the resulting dataframe.
        :type drop_raw: boolean.
    
    Methods:
        "fit_transform": applies the logarithmic transformation over a provided dataset.
        "log_func": static method for calculating the natural logarithm of a given real valued number. Note that negative values
        are returned as zero after the transformation.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, to_log, eps: float = 0.0001, drop_raw: bool = True):
        self.to_log = to_log
        self.eps = eps
        self.drop_raw = drop_raw
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:        
        # Consistency check:
        if len([v for v in self.to_log if v not in list(data.columns)]) > 0:
            not_found = [v for v in self.to_log if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        transf_data = data.copy()
        
        # Loop over selected variables:
        for v in self.to_log:
            transf_data[f'L#{v}'] = [self.log_func(x) for x in transf_data[v]]

            if self.drop_raw:
                transf_data.drop(v, axis=1, inplace=True)

        return transf_data

    @staticmethod
    def log_func(x, eps: float = 0.0001) -> float:
        """
        Function for calculating the natural logarithm of a given real valued number.

        Negative values are returned as zero after the transformation.

        :param eps: minimal value to be added to zero valued observations in order to correctly calculate the logarithm.
        :type eps: float.

        :return: log-transformed value.
        :rtype: float.
        """
        if x < 0:
            return 0
        else:
            return np.log(x + eps)

####################################################################################################################################
# Class for applying scale transformation over numerical variables:

class ScaleNumericalVars:
    """
    Class for applying scale transformation over numerical variables.

    Initialization attributes:
        :param to_scale: names of variables that are going to be standard scaled.
        :type to_scale: list.
        :param which_scale: indicates whether standard scale or min-max scale transformation should be applied. So, choose among
        ["standard_scale", "min_max_scale"].
        :type which_scale: boolean.
        :param scale: scale of transformed data.
        :type scale: float.

    Methods:
        "fit": prepares the chosen scale transformation from a training data of reference. This means calculating the necessary
        statistics for scaling.
        "transform": applies the scale transformation over a data passed as argument of the method. Can be either training data or
        an independent dataset.
        "standardize_stats": static method that calculates statistics for standard scaling numerical data.
        "standardize_data": static method for standard scaling numerical data.
        "min_max_stats": static method that produces the statistics for min-max scaling numerical data.
        "min_max_scale": static method that applies the min-max scaling of numerical variables.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, to_scale: list, which_scale: str ="standard_scale", scale: float = 1.0):
        self.to_scale = to_scale
        self.which_scale = which_scale
        self.scale = scale
    
    def fit(self, training_data):
        # Consistency check:
        if len([v for v in self.to_scale if v not in list(training_data.columns)]) > 0:
            not_found = [v for v in self.to_scale if v not in list(training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        if self.which_scale=="standard_scale":
            self.stats = self.standardize_stats(training_data=training_data, to_scale=self.to_scale)
        
        elif self.which_scale=="min_max_scale":
            self.stats = self.min_max_stats(data=training_data, to_scale=self.to_scale, scale=self.scale)
        
        else:
            raise ValueError('Please, define "which_scale" initialization argument as either "standard_scale" or "min_max_scale".')

    def transform(self, data) -> pd.DataFrame:
        # Consistency check:
        if len([v for v in self.to_scale if v not in list(data.columns)]) > 0:
            not_found = [v for v in self.to_scale if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        if self.which_scale=="standard_scale":
            return self.standardize_data(data=data, stats=self.stats)

        elif self.which_scale=="min_max_scale":
            return self.min_max_scale(data=data, stats=self.stats)
        
        else:
            return
  
    @staticmethod
    def standardize_stats(training_data, to_scale) -> dict:
        """
        Function that calculates statistics for standard scaling numerical data.
        
        :param training_data: data from which statistics should be calculated.
        :type training_data: dataframe.
        :param to_scale: names of variables that are going to be standard scaled.
        :type to_scale: list.
        
        :return: means and standard deviations in dictionaries whose keys are names of variables and values are the
        corresponding values of statistics.
        :rtype: dictionary.
        """
        means = dict(training_data[[c for c in training_data.columns if c in to_scale]].mean())
        stds = dict(training_data[[c for c in training_data.columns if c in to_scale]].std())
        
        return {'means': means, 'stds': stds}

    @staticmethod
    def standardize_data(data, stats) -> pd.DataFrame:
        """
        Function for standard scaling numerical data.
        
        :param data: data whose numerical variables should be standard scaled.
        :type data: dataframe.
        :param stats: means and standard deviations in dictionaries whose keys are names of variables and values are
        the corresponding values of statistics.
        :type stats: dictionary.
        
        :return: data with numerical variables standard scaled.
        :rtype: dataframe.
        """
        standardized_data = data.copy()
        
        for k in stats['means']:
            standardized_data[k] = standardized_data[k].apply(lambda x: (x-stats['means'][k])/stats['stds'][k])
            
        return standardized_data

    @staticmethod
    def min_max_stats(data: pd.DataFrame, to_scale: list, scale: float = 1) -> dict:
        """
        Function that produces the statistics for min-max scaling numerical data.

        :param data: dataframe of reference for calculating the values used during min-max scaling.
        :type data: pandas dataframe.
        :param to_scale: names of columns that should be min-max scaled.
        :type to_scale: list.
        :param scale: scale of transformed data.
        :type scale: float.

        :return: values used during min-max scaling for each variable.
        :rtype: dictionary.
        """
        scaled_data = data.copy()
        min_max_stats_ = {}

        for v in to_scale:
            min_ref = scaled_data[v].min()
            scaled_data[v] = scaled_data[v].apply(lambda x: x - min_ref)
            max_ref = scaled_data[v].max()

            min_max_stats_[v] = (min_ref, max_ref, scale)

        return min_max_stats_

    @staticmethod
    def min_max_scale(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
        """
        Function that applies the min-max scaling of numerical variables.

        :param data: dataset to be transformed.
        :type data: pandas dataframe.
        :param stats: values used during min-max scaling for each variable.
        :type stats: dictionary.

        :return: transformed data.
        :rtype: pandas dataframe.
        """
        scaled_data = data.copy()

        for v in stats:
            min_ref = stats[v][0]
            max_ref = stats[v][1]
            scale = stats[v][2]

            scaled_data[v] = scaled_data[v].apply(lambda x: (x - min_ref)*scale/max_ref)

        return scaled_data

####################################################################################################################################
# Class that creates dummies from categorical features following a variance criterium for selecting categories:

class OneHotEncoding:
    """
    Arguments for initialization:
        :param categorical_features: list of categorical features whose categories should be selected.
        :type categorical_features: list.
        :param variance_param: parameter for selection based on the variance of a given dummy variable.
        :type variance_param: float.
        :param clean_text: indicates whether texts of categories should be cleaned before creating dummy variables from them.
        :type clean_text: boolean.
    
    Methods:
        "fit": method for finding which categories should be preserved in order to create dummy variables from them.
        "transform": applies the one-hot encoding transformation according to selected categories.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, categorical_features: list,  variance_param: float = 0.01, clean_text: bool = True):
        self.categorical_features = categorical_features
        self.variance_param = variance_param
        self.clean_text = clean_text
        

    def fit(self, training_data: pd.DataFrame):
        transf_training_data = training_data.copy()

        # Consistency check:
        if len([v for v in self.categorical_features if v not in list(transf_training_data.columns)]) > 0:
            not_found = [v for v in self.categorical_features if v not in list(transf_training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        self.selected_cat, self.categories_assessment = {}, {}

        # Loop over categorical features:
        for f in self.categorical_features:
            if self.clean_text:
                # Treating texts:
                transf_training_data[f] = transf_training_data[f].apply(text_clean)

            # Creating dummy variables:
            dummies_cat = pd.get_dummies(transf_training_data[f]) 
            dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

            # Selecting dummies_cat depending on their variance:
            self.selected_cat[f] = [d for d in dummies_cat.columns if dummies_cat[d].var() > self.variance_param]

            # Assessing categories:
            self.categories_assessment[f] = {
                "num_categories": len(dummies_cat.columns),
                "num_selected_categories": len(self.selected_cat[f]),
                "selected_categories": self.selected_cat[f]
            }

    def transform(self, data: pd.DataFrame, treat_missings: bool = True, drop_raw: bool = True,
                  verbose: bool = False) -> pd.DataFrame:
        # Consistency check:
        if len([v for v in self.categorical_features if v not in list(data.columns)]) > 0:
            not_found = [v for v in self.categorical_features if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        transf_data = data.copy()

        # Loop over categorical features:
        for f in self.categorical_features:
            if treat_missings:
                transf_data[f].fillna('missing_value', inplace=True)

            if self.clean_text:
                # Treating texts:
                transf_data[f] = transf_data[f].apply(text_clean)

            # Creating the dummy variables:
            dummies_cat = pd.get_dummies(transf_data[f])
            dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

            # Checking if all categories selected from training data also exist for test data:
            for c in [c for c in self.selected_cat[f] if c not in dummies_cat.columns]:
                dummies_cat[c] = [0 for i in range(len(dummies_cat))]

            # Preserving only selected categories:
            for c in [c for c in dummies_cat.columns if c not in self.selected_cat[f]]:
                dummies_cat.drop(c, axis=1, inplace=True)

        if verbose:
            print(f'\033[1mNumber of categorical features:\033[0m {len(self.categorical_features)}')
            print(f'\033[1mNumber of overall selected dummies:\033[0m {dummies_cat.shape[1]}.')

        transf_data = pd.concat([transf_data, dummies_cat], axis=1, sort=False)

        if drop_raw:
            transf_data.drop(self.categorical_features, axis=1, inplace=True)
        
        return transf_data

####################################################################################################################################
# Class for handling missing values:

class TreatMissings:
    """
    Class for handling missing values.

    Arguments for initialization:
        :param vars_to_treat: names of (numerical) variables whose missings should be treated. If None is provided, then variables
        to be treated will be inferred from data when "fit_transform" is applied.
        :type vars_to_treat: list or None.
        :param drop_vars: names of support variables, i.e., those that are not subject to direct data analysis.
        :type drop_vars: list.
        :param cat_vars: names of categorical variables.
        :type cat_vars: list.
        :param method: indicates how missing values should be treated. Choose among ['create_binary', 'impute_stat'].
        :type method: string.
        :param statistic: declares which statistic should be used for missing values treatment. Choose among ["mean", "median"].
        :type statistic: string.
        :param treat_remaining: indicates whether remaining missing values should be treated after the chosen treatment method.
        :type treat_remaining: boolean.
    
    Methods:
        "fit_transform": applies the selected missing values treament over the provided data.
        "create_binary": static method for missing values treatment under which 0 is imputted and a binary variable is created to
        indicate whether a value is missing.
        "calculate_stat": static method for calculating statistics that can be used for missing values imputation.
        "impute_stat": static method for imputting a value of a statistic on missings.
        "categorical_missings": static method for creating a category named "missing_value" for missing values of categorical
        variables.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, drop_vars: list, cat_vars: list, method: str = 'create_binary', statistic: str = None,
                 treat_remaining: bool = True, vars_to_treat: Optional[list] = None):
        self.vars_to_treat = vars_to_treat
        self.method = method
        self.drop_vars = drop_vars
        self.cat_vars = cat_vars
        self.statistic = statistic
        self.treat_remaining = treat_remaining
    
    def fit(self, training_data: pd.DataFrame = None):
        """
        :param data: dataset for calculating statistics in the imputation method.
        :type data: dataframe.

        :return: fitted object for treating missing values.
        :rtype: None.
        """
        if self.vars_to_treat is None:
            self.vars_to_treat = [c for c in list(training_data.columns) if (c not in self.drop_vars) & (c not in self.cat_vars) &
                                  (training_data[c].isnull().sum() > 0)]

        if self.method=='impute_stat':
            self._stats = self.calculate_stat(
                training_data=training_data, vars_to_treat=self.vars_to_treat, drop_vars=self.drop_vars,
                cat_vars=self.cat_vars, statistic=self.statistic
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: dataset whose missing values should be treated.
        :type data: dataframe.

        :return: dataframe with treated missing values.
        :rtype: dataframe.
        """
        transf_data = data.copy()

        if self.method=='create_binary':
            transf_data = self.create_binary(data=transf_data, vars_to_treat=self.vars_to_treat, drop_vars=self.drop_vars,
                                             cat_vars=self.cat_vars)

        elif self.method=='impute_stat':
            transf_data = self.impute_stat(data=transf_data, stats=self._stats)

        else:
            raise ValueError('Please, define "method" initialization argument as either "create_binary" or "impute_stat".')

        # Treating missings of categorical variables:
        cat_vars_to_treat = [c for c in list(transf_data.columns) if (c in self.cat_vars) & (transf_data[c].isnull().sum() > 0)]
        transf_data = self.categorical_missings(data=transf_data, cat_vars=cat_vars_to_treat)

        # Treating remaining missings:
        if self.treat_remaining:
            for v in [c for c in list(transf_data.columns) if (c not in self.drop_vars) & (transf_data[c].isnull().sum() > 0)]:
                transf_data[v].fillna(0, inplace=True)

        return transf_data

    @staticmethod
    def create_binary(data: pd.DataFrame, vars_to_treat: list, drop_vars: list, cat_vars: list) -> pd.DataFrame:
        """
        Missing values treatment under which 0 is imputted and a binary variable is created to indicate whether a value is missing.

        :param data: dataset whose missing values should be treated.
        :type data: dataframe.
        :param vars_to_treat: names of variables whose missings should be treated.
        :type vars_to_treat: list.
        :param drop_vars: names of support variables, i.e., those that are not subject to direct data analysis.
        :type drop_vars: list.
        :param cat_vars: names of categorical variables.
        :type cat_vars: list.

        :return: dataset with treated missing values.
        :rtype: dataframe.
        """
        # Consistency check:
        if len([v for v in vars_to_treat if v not in list(data.columns)]) > 0:
            not_found = [v for v in vars_to_treat if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        transf_data = data.copy()

        # Loop over variables:
        for v in vars_to_treat:
            # Creating a binary variable that indicates whether a value is missing:
            transf_data[f'NA#{v}'] = transf_data[v].apply(lambda x: 1 if pd.isna(x) else 0)
            
            # Imputing missing values:
            transf_data[v].fillna(0, inplace=True)
        
        return transf_data

    @staticmethod
    def calculate_stat(training_data: pd.DataFrame, vars_to_treat: list, drop_vars: list, cat_vars: list,
                       statistic: str = 'mean') -> pd.DataFrame:
        """
        Function that calculates statistics for missing values treatment.

        :param training_data: dataset from which statistics should be calculated for missing values imputation.
        :type training_data: dataframe.
        :param vars_to_treat: names of variables whose missings should be treated.
        :type vars_to_treat: list.
        :param drop_vars: names of support variables, i.e., those that are not subject to direct data analysis.
        :type drop_vars: list.
        :param cat_vars: names of categorical variables.
        :type cat_vars: list.
        :param statistic: declares which statistic should be used for missing values treatment. Choose among ["mean", "median"].
        :type statistic: string.

        :return: statistics for each variable whose missings should be treated.
        :rtype: dictionary.
        """
        # Consistency check:
        if len([v for v in vars_to_treat if v not in list(training_data.columns)]) > 0:
            not_found = [v for v in vars_to_treat if v not in list(training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        stats = {}

        # Loop over variables:
        for v in vars_to_treat:
            if statistic == 'mean':
                stats[v] = training_data[v].mean()

            elif statistic == 'median':
                stats[v] = training_data[v].median()

        return stats

    @staticmethod
    def impute_stat(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
        """
        Missing values treatment under which a statistic is imputted.

        :param data: dataset whose missing values should be treated.
        :type data: dataframe.
        :param stats: value to be imputted for each variable.
        :type stats: dictionary.

        :return: statistics for each variable whose missings should be treated.
        :rtype: dictionary.
        """
        # Consistency check:
        if len([v for v in list(stats.keys()) if v not in list(data.columns)]) > 0:
            not_found = [v for v in list(stats.keys()) if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        transf_data = data.copy()

        # Loop over variables:
        for v in stats:
            # Imputing missing values:
            transf_data[v].fillna(stats[v], inplace=True)
        
        return transf_data

    @staticmethod
    def categorical_missings(data: pd.DataFrame, cat_vars: list) -> pd.DataFrame:
        """
        Missing values treatment for categorical data.

        :param data: dataset whose missing values should be treated.
        :type data: dataframe.
        :param cat_vars: names of categorical variables.
        :type cat_vars: list.

        :return: dataset with treated missing values.
        :rtype: dataframe.
        """
        # Consistency check:
        if len([v for v in cat_vars if v not in list(data.columns)]) > 0:
            not_found = [v for v in cat_vars if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        transf_data = data.copy()

        # Loop over categorical variables:
        for v in cat_vars:
            transf_data[v].fillna('missing_value', inplace=True)
        
        return transf_data

####################################################################################################################################
# Class for handling with outliers in numerical data:

class OutliersTreat:
    """
    Class for handling with outliers in numerical data.

    Arguments for initialization:
        :param vars_to_treat: names of (numerical) variables whose outliers should be treated.
        :type vars_to_treat: list.
        :param method: name of the outliers treatment method. Choose among ["quantile", "iqr"].
        :type method: string.
        :param quantile: value of quantile below which (or above 1 - quantile) a value is considered a lower (upper) outlier.
        :type quantile: float.
        :param k: parameter for the interquartile range (IQR) calculation.
        :type k: float.

    Methods:
        "fit": calculate statistics for outliers handling.
        "transform": performs the outliers treatment.
        "quantile_calculation": static method for calculating quantiles, quantile and 1 - quantile.
        "quantile_imputation": static method for imputing quantiles for outlier values.
        "iqr": static method that calculates the lower and upper values of the interquartile range (IQR).
        "iqr_calculation": static method that calculates IQR for provided variables given training data.
        "iqr_imputation": static method that replaces outlier values for IQR lower or upper values.
        ""
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, vars_to_treat: list, method: str = 'quantile', quantile: float = 0.025, k: float = 1.5):
        self.vars_to_treat = vars_to_treat
        self.method = method
        self.quantile = quantile
        self.k = k
    
    def fit(self, training_data: pd.DataFrame):
        # Consistency check:
        if len([v for v in self.vars_to_treat if v not in list(training_data.columns)]) > 0:
            not_found = [v for v in self.vars_to_treat if v not in list(training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        if self.method == 'quantile':
            self.quant = self.quantile_calculation(training_data=training_data, vars_to_treat=self.vars_to_treat,
                                                   quantile=self.quantile)
        
        elif self.method == 'iqr':
            self.iqr_dict = self.iqr_calculation(training_data=training_data, vars_to_treat=self.vars_to_treat, k=self.k)
        
        else:
            raise ValueError('Please, define "method" initialization argument as either "quantile" or "iqr".')

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Consistency check:
        if len([v for v in self.vars_to_treat if v not in list(data.columns)]) > 0:
            not_found = [v for v in self.vars_to_treat if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")
          
        if self.method == 'quantile':
            return self.quantile_imputation(data=data, quant=self.quant)
        
        elif self.method == 'iqr':
            return self.iqr_imputation(data=data, iqr_dict=self.iqr_dict)
        
        else:
            return

    @staticmethod
    def quantile_calculation(training_data: pd.DataFrame, vars_to_treat: list, quantile: float = 0.025) -> dict:
        """
        Function for calculating quantiles, quantile and 1 - quantile.

        :param training_data: data of reference for calculating quantiles.
        :type training_data: dataframe.
        :param vars_to_treat: names of variables for outliers treatment.
        :type vars_to_treat: list.
        :param quantile: value of quantile below which (or above 1 - quantile) a value is considered a lower (upper) outlier.
        :type quantile: float.

        :return: pairs of quantiles for each required variable.
        :rtype: dictionary.
        """
        # Consistency check:
        if len([v for v in vars_to_treat if v not in list(training_data.columns)]) > 0:
            not_found = [v for v in vars_to_treat if v not in list(training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        quant = {}

        # Loop over variables to treat:
        for v in vars_to_treat:
            quant[v] = (training_data[v].quantile(q=quantile), training_data[v].quantile(q=1-quantile))
        
        return quant
    
    @staticmethod
    def quantile_imputation(data: pd.DataFrame, quant: dict) -> pd.DataFrame:
        """
        Function for imputing quantiles for outlier values.

        :param data: data for replacing outliers for quantiles.
        :type data: dataframe.
        :param quant: lower and upper quantiles for each required variable.
        :type quant: dictionary.

        :return: data with treated outliers.
        :rtype: dataframe.
        """
        # Consistency check:
        if len([v for v in list(quant.keys()) if v not in list(data.columns)]) > 0:
            not_found = [v for v in list(quant.keys()) if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")
        
        transf_data = data.copy()

        # Loop over variables to treat:
        for v in quant:
            transf_data[v] = transf_data[v].apply(lambda x: quant[v][1] if x > quant[v][1] else
                                                  (quant[v][0] if x < quant[v][0] else x))

        return transf_data

    @staticmethod
    def iqr(numerical_data: Union[list, tuple, pd.Series], k: float = 1.5) -> tuple:
        """
        Function that calculates the lower and upper values of the interquartile range (IQR).

        :param numerical_data: collection of values of a numerical variable.
        :type numerical_data: list, or tuple, or pandas series.
        :param k: parameter for the interquartile range (IQR) calculation.
        :type k: float.

        :return: lower and upper bound of IQR.
        :rtype: tuple.
        """
        q1 = np.quantile(numerical_data, q=0.25)
        q3 = np.quantile(numerical_data, q=0.75)

        return q1 - k*(q3 - q1), q3 + k*(q3 - q1)

    @staticmethod
    def iqr_calculation(training_data: pd.DataFrame, vars_to_treat: list, k: float = 1.5) -> dict:
        """
        Function that calculates IQR for provided variables given training data.

        :param training_data: data of reference for calculating IQR of numerical variables.
        :type training_data: dataframe.
        :param vars_to_treat: names of variables for outliers treatment.
        :type vars_to_treat: list.
        :param k: parameter for the interquartile range (IQR) calculation.
        :type k: float.

        :return: lower and upper bound of IQR for each required variable.
        :rtype: dictionary.
        """
        # Consistency check:
        if len([v for v in vars_to_treat if v not in list(training_data.columns)]) > 0:
            not_found = [v for v in vars_to_treat if v not in list(training_data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")

        iqr_dict = {}

        # Loop over variables to treat:
        for v in vars_to_treat:
            iqr_dict[v] = OutliersTreat.iqr(training_data[v], k=k)
        
        return iqr_dict
    
    @staticmethod
    def iqr_imputation(data: pd.DataFrame, iqr_dict: dict) -> pd.DataFrame:
        """
        Function that replaces outlier values for IQR lower or upper values.

        :param data: data for treating outliers.
        :type data: dataframe.
        :param iqr_dict: lower and upper bound of IQR for each required variable.
        :type iqr_dict: dictionary.

        :return: data with treated outliers.
        :rtype: dataframe.
        """
        # Consistency check:
        if len([v for v in list(iqr_dict.keys()) if v not in list(data.columns)]) > 0:
            not_found = [v for v in list(iqr_dict.keys()) if v not in list(data.columns)]
            raise IndexError(f"Variables {not_found} do not exist in provided data.")
        
        transf_data = data.copy()

        # Loop over variables to treat:
        for v in iqr_dict:
            transf_data[v] = transf_data[v].apply(lambda x: iqr_dict[v][1] if x > iqr_dict[v][1] else
                                                  (iqr_dict[v][0] if x < iqr_dict[v][0] else x))
        
        return transf_data

####################################################################################################################################
# Class for sequentially applying data transformations:

class Pipeline:
    """
    Class for sequentially applying data transformations.

    Initialization arguments:
        :param operations: collection of instances of classes that implement data transformations. These instances must have either
        a pair of "fit" and "transform" methods, or at least a "fit_transform" unified method.
        :type operations: list or tuple.

    Methods:
        "transform": applies the data transformations declared in the list of operations for a training data and a collection of
        additional data (validation, test, application, etc.).
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()
    
    def __init__(self, operations: Union[list, tuple]):
        self.operations = operations

    def transform(self, data_list: Union[List[pd.DataFrame], Tuple[pd.DataFrame]], training_data: pd.DataFrame) -> pd.DataFrame:
        transf_training_data = training_data.copy()
        transf_data_list = [data.copy() for data in data_list]
        
        # Loop over data transformations:
        for op in self.operations:
            if 'fit' not in dir(op):
                transf_training_data = op.fit_transform(data=transf_training_data)

                # Loop over application data:
                for i in range(len(transf_data_list)):
                    transf_data_list[i] = op.fit_transform(data=transf_data_list[i])

            else:
                op.fit(training_data=transf_training_data)
                transf_training_data = op.transform(data=transf_training_data)

                # Loop over application data:
                for i in range(len(transf_data_list)):
                    transf_data_list[i] = op.transform(data=transf_data_list[i])

        return transf_training_data, transf_data_list
