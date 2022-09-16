####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
from typing import Union

# pip install unidecode
from unidecode import unidecode

# pip install plotly==4.6.0
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that splits data into train and test set:

def train_test_split(dataframe, preserve_date=False, date_var='date', test_ratio=0.5, shuffle=False, seed=None,
                     date_split=False, last_train=None):
    """
    Function that splits data into train and test set.
    
    :param dataframe: complete set of data.
    :type dataframe: dataframe.
    :param preserve_date: indicates whether to perform split based on volume of data, but not mingling instances
    from the same date.
    :type preserve_date: boolean.
    :param date_var: name of the date variable to consider during the split.
    :type date_var: string.
    :param seed: seed for shuffle.
    :type seed: integer.
    :param test_ratio: proportion of data to be allocated into test set.
    :type test_ratio: float.
    :param shuffle: indicates whether to shuffle data previously to the split.
    :type shuffle: boolean.
    :param date_split: indicates whether the train-test split should consider the date as reference instead of data volume.
    :type date_split: boolean.
    :param last_train: last period present in the training data.
    :type last_train: string.
    
    :return: training and test dataframes.
    :rtype: tuple.
    """
    df = dataframe.copy()
    df.reset_index(drop=True, inplace=True)
    
    if shuffle:
        df = df.sample(len(df), random_state=seed)
    
    if date_split:
        # Train-test split:
        df_train = df[df[date_var]<=last_train]
        df_test = df[df[date_var]>last_train]
    
    else:
        if preserve_date:
            # Number of instances by date:
            orders_by_date = pd.DataFrame(data={
                'date': df[date_var].apply(lambda x: x.date()).value_counts().index,
                'freq': df[date_var].apply(lambda x: x.date()).value_counts().values}).sort_values('date')

            # Accumulated number of instances by date:
            orders_by_date['acum'] = np.cumsum(orders_by_date.freq)
            orders_by_date['acum_share'] = [a/orders_by_date['acum'].max() for a in orders_by_date['acum']]

            # Date gathering 1 - test_ratio of data:
            last_date_train = orders_by_date.iloc[np.argmin(abs(orders_by_date['acum_share'] - (1 - test_ratio)))]['date']

            # Train-test split:
            df_test = df[df[date_var].apply(lambda x: x.date()) > last_date_train]
            df_train = df[df[date_var].apply(lambda x: x.date()) <= last_date_train]

        else:
            # Indexes for training and test data:
            test_indexes = [True if i > int(len(df)*(1 - test_ratio)) else False for i in range(len(df))]
            train_indexes = [True if i==False else False for i in test_indexes]

            # Train-test split:
            df_train = df.iloc[train_indexes, :]
            df_test = df.iloc[test_indexes, :]
    
    return (df_train, df_test)

####################################################################################################################################
# Function for cleaning texts:

def text_clean(text, language='portuguese', remove_accent=True, remove_extra_spaces=True, remove_spaces=True, replace_spaces=True,
               delete_signs=True, lower=True):
    """
    Function for cleaning texts.

    :param text: text to be cleaned.
    :type text: string.
    :param language: expected language in which the text was written.
    :type language: string.
    :param remove_accent: indicates whether to remove accents.
    :type remove_accent: boolean.
    :param remove_extra_spaces: indicates whether to remove extra spaces.
    :type remove_extra_spaces: boolean.
    :param remove_spaces: indicates whether to remove spaces before and after texts.
    :type remove_spaces: boolean.
    :param replace_spaces: indicates whether to replace spaces for "_".
    :type replace_spaces: boolean.
    :param delete_signs: indicates whether to remove signs.
    :type delete_signs: boolean.
    :param lower: indicates whether to define the string as lower case.
    :type lower: boolean.

    :return: text cleaned after the some of the following operations: removal of accent, extra spaces, ponctuation, stop words, and
    eventually of upper case letters.
    :rtype: string.
    """

    if pd.isnull(text):
        return text

    else:
        text_cleaned = str(text)

        # Removing accent:
        if remove_accent:
            text_cleaned = unidecode(text)
        else:
            text_cleaned = text

        # Removing extra spaces:
        if remove_extra_spaces:
            text_cleaned = re.sub(' +', ' ', text_cleaned)

        # Removing spaces before and after text:
        if remove_spaces:
            text_cleaned = str.strip(text_cleaned)

        # Replaces spaces for "_":
        if replace_spaces:
            text_cleaned = text_cleaned.replace(' ', '_')

        # Removing stop words:
        if delete_signs:
            for m in '.,;+-!@#$%¨&*()[]{}\\/|':
                if m in text_cleaned:
                    text_cleaned = text_cleaned.replace(m, '')

        # Setting text to lower case:
        if lower:
            text_cleaned = text_cleaned.lower()

        return text_cleaned

####################################################################################################################################
# Function that produces a dataframe with frequency of features by class and returns lists with features names by class:

def classify_variables(dataframe, vars_to_drop=[], drop_excessive_miss=True, excessive_miss=0.95,
                       drop_no_var=True, minimum_var=0, validation_data=None, test_data=None):
    """
    Function that produces a dataframe with frequency of features by class and returns lists with features names by class.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :param vars_to_drop: list of support columns.
    :type vars_to_drop: list.

    :param drop_excessive_miss: flag indicating whether columns with excessive missings should be dropped out.
    :type drop_excessive_miss: boolean.

    :param excessive_miss: share of missings above which columns are dropped from the dataframes.
    :type excessive_miss: float.

    :param drop_no_var: flag indicating whether columns with no variance should be dropped out.
    :type drop_no_var: boolean.

    :param minimum_var: value of variance below which columns are dropped from the dataframes.
    :type minimum_var: float.

    :param validation_data: additional data.
    :type validation_data: dataframe.

    :param test_data: additional data.
    :type test_data: dataframe.

    :return: dataframe and lists with features by class.
    :rtype: dictionary.
    """
    print(f'Initial number of features: {dataframe.drop(vars_to_drop, axis=1).shape[1]}.')

    if drop_excessive_miss:
        # Dropping features with more than a predefined rate of missings in the training data:
        excessive_miss_train = [c for c in dataframe.drop(vars_to_drop, axis=1) if
                                sum(dataframe[c].isnull())/len(dataframe) > excessive_miss]

        if len(excessive_miss_train) > 0:
            dataframe.drop(excessive_miss_train, axis=1, inplace=True)

            if validation_data is not None:
                validation_data.drop(excessive_miss_train, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(excessive_miss_train, axis=1, inplace=True)

        print(f'{len(excessive_miss_train)} features were dropped for excessive number of missings!')
        
    # Classifying features:
    feature_class = pd.DataFrame(data={
        'feature': [c for c in dataframe.columns if c not in vars_to_drop],
        'n_unique': [dataframe[c].nunique() for c in dataframe.columns if c not in
                     vars_to_drop],
        'd_type': [dataframe.dtypes[c] for c in dataframe.columns if c not in
                   vars_to_drop],
    })

    binary_vars = list(feature_class[(feature_class.n_unique==2) & ~(feature_class.d_type==object)]['feature'])
    cont_vars = list(feature_class[(feature_class.n_unique>2) & ~(feature_class.d_type==object)]['feature'])
    cat_vars = list(feature_class[feature_class.d_type==object]['feature'])
        
    if drop_no_var:
        # Dropping features with no variance in the training data:
        no_variance = [c for c in dataframe.drop(vars_to_drop, axis=1).drop(cat_vars,
                                                                            axis=1) if np.nanvar(dataframe[c])<=minimum_var]

        if len(no_variance) > 0:
            dataframe.drop(no_variance, axis=1, inplace=True)
            if validation_data is not None:
                validation_data.drop(no_variance, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(no_variance, axis=1, inplace=True)

        print(f'{len(no_variance)} features were dropped for having no variance!')
        
    print(f'{dataframe.drop(vars_to_drop, axis=1).shape[1]} remaining features.')
    print('\n')
    
    # Dataframe presenting the frequency of features by class:
    feats_assess = pd.DataFrame(data={
        'class': ['cat_vars', 'binary_vars', 'cont_vars', 'vars_to_drop'],
        'frequency': [len(cat_vars), len(binary_vars), len(cont_vars), len(vars_to_drop)]
    })
    feats_assess.sort_values('frequency', ascending=False, inplace=True)
    
    # Dictionary with outputs from the function:
    feats_assess_dict = {
        'feats_assess': feats_assess,
        'cat_vars': cat_vars,
        'binary_vars': binary_vars,
        'cont_vars': cont_vars
    }
    
    if drop_excessive_miss:
        feats_assess_dict['excessive_miss_train'] = excessive_miss_train

    if drop_no_var:
        feats_assess_dict['no_variance'] = no_variance
    
    return feats_assess_dict

####################################################################################################################################
# Function that produces an assessment of the occurrence of missing values:

def assessing_missings(dataframe):
    """
    Function that produces an assessment of the occurrence of missing values.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :return: dataframe with frequency and share of missings by feature.
    :rtype: dataframe.
    """
    # Dataframe with the number of missings by feature:
    missings_dict = dataframe.isnull().sum().sort_values(ascending=False).to_dict()

    missings_df = pd.DataFrame(data={
        'feature': list(missings_dict.keys()),
        'missings': list(missings_dict.values()),
        'share': [m/len(dataframe) for m in list(missings_dict.values())]
    })

    print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_df.missings > 0)) +
          ' out of {} features'.format(len(missings_df)) +
          ' ({}%).'.format(round((sum(missings_df.missings > 0)/len(missings_df))*100, 2)))
    print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_df.missings.mean())) +
          ' out of {} observations'.format(len(dataframe)) +
          ' ({}%).'.format(round((int(missings_df.missings.mean())/len(dataframe))*100,2)))
    
    return missings_df

####################################################################################################################################
# Function that assess the number of missings in a dataframe:

def missings_detection(dataframe, name='df', var=None):
    """"
    Function that assess the number of missings in a dataframe

    :param dataframe: dataframe for which missings should be detected.
    :type dataframe: dataframe.
    
    :param name: name of dataframe.
    :type name: string.
    
    :param var: name of variable whose missings should be detected (optional).
    :type var: string.

    :return: prints the number of missings when there is a positive amount of them.
    """

    if var:
        num_miss = dataframe[var].isnull().sum()
        if num_miss > 0:
            print(f'Problem - There are {num_miss} missings for "{var}" in dataframe {name}.')

    else:
        num_miss = dataframe.isnull().sum().sum()
        if num_miss > 0:
            print(f'Problem - Number of overall missings detected in dataframe {name}: {num_miss}.')

####################################################################################################################################
# Function that forces consistency between reference (training) and additional (validation, test) data:

def data_consistency(dataframe, verbose=True, **kwargs):
    """
    Function that forces consistency between reference (training) and additional (validation, test) data:

    The keyword arguments are expected to be dataframes whose argument names indicate the nature of the passed data. For instance,
    'test_data=df_test' would be a dataframe with test instances.

    :param dataframe: reference data.
    :type dataframe: dataframe.
    :param verbose: indicates whether the result of consistency check should be printed.
    :type verbose: boolean.

    :return: dataframes with consistent data.
    :rtype: dictionary.
    """
    consistent_data = {}
    
    for d in kwargs.keys():
        consistent_data[d] = kwargs[d].copy()
        
        # Columns absent in reference data:
        absent_train = [c for c in kwargs[d].columns if c not in dataframe.columns]
        
        # Columns absent in additional data:
        absent_test = [c for c in dataframe.columns if c not in kwargs[d].columns]
        
        # Creating absent columns:
        for c in absent_test:
            consistent_data[d][c] = 0
    
        # Preserving consistency between reference and additional data:
        consistent_data[d] = consistent_data[d][dataframe.columns]
        
        # Checking consistency:
        if sum([1 for r, a in zip(dataframe.columns, consistent_data[d].columns) if r != a]):
            print('Problem - Reference and additional datasets are inconsistent!')
        else:
            print(f'Training and {d.replace("_", " ")} are consistent with each other.')
    
    return consistent_data

####################################################################################################################################
# Function that returns the amount of time for running a block of code:

def running_time(start_time, end_time, print_time=True):
    """
    Function that returns the amount of time for running a block of code.
    
    :param start_time: time point when the code was initialized.
    :type start_time: datetime object obtained by executing "datetime.now()".

    :param end_time: time point when the code stopped its execution.
    :type end_time: datetime object obtained by executing "datetime.now()".

    :param print_unit: unit of time for presenting the runnning time.
    :type print_unit: string.
    
    :return: prints start, end time and running times, besides of returning running time in seconds.
    :rtype: integer.
    """
    if print_time:
        print('------------------------------------')
        print('\033[1mRunning time:\033[0m ' + str(round(((end_time - start_time).total_seconds())/60, 2)) +
              ' minutes.')
        print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
        print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
        print('------------------------------------')
    
    return (end_time - start_time).total_seconds()

####################################################################################################################################
# Function that calculates cross-entropy given true labels and predictions:

def cross_entropy_loss(y_true, p):
    prediction = np.clip(p, 1e-14, 1. - 1e-14)
    return -np.sum(y_true*np.log(prediction) + (1-y_true)*np.log(1-prediction))/len(y_true)

####################################################################################################################################
# Function that produces a random sample preserving classes distribution of a categorical variable:

def representative_sample(dataframe, categorical_var, classes, sample_share=0.5):
    """
    Arguments:
        'dataframe': dataframe containing indices to be drawn and a categorical variable whose distribution in
        the sample should be kept equal to that of whole data.
        'categorical_var': categorical variable of reference (string).
        'classes': dictionary whose keys are classes and values are their shares in the entire data.
        'sample_share': float indicating sample length as the proportion of entire data length.
    Output:
        Returns a list with randomly picked indices.
    """
    
    # Randomly picked indices:
    samples = [sorted(np.random.choice(dataframe[dataframe[categorical_var]==k].index,
                                       size=int(classes[k]*sample_share*len(dataframe)),
                                       replace=False)) for k in classes.keys()]
    
    sample = []

    # Loop over samples:
    for l in samples:
        # Loop over indices:
        for i in l:
            sample.append(i)
    
    return sample

####################################################################################################################################
# Function that calculates frequencies of elements in a list:

def frequency_list(_list):
    """
    Function that calculates frequencies of elements in a list:
    
    :param _list: .
    :type _list: list.
    
    :return: dictionary whose keys are the elements in a list and values are their frequencies.
    :rtype: dictionary.
    """
    _set = set(_list)
    freq_dict = {}

    # Loop over unique elements:
    for f in _set:
        freq_dict[f] = 0

        # Counting frequency:
        for i in _list:
            if i == f:
                freq_dict[f] += 1
    
    return freq_dict

####################################################################################################################################
# Function that checks the existence of duplicated instances:

def check_duplicates(dataframe, primary_key=None):
    """
    Function that checks the existence of duplicated instances.

    :param dataframe: data to be assessed.
    :type dataframe: dataframe.
    :param primary_key: list with names of variables which points to unique instances.
    :type primary_key: list.

    :return: boolean indicating whether there are duplicated instances, besides of the printing of the number of duplicated rows.
    :rtype: boolean.
    """
    num_duplicated = len(dataframe) - len(dataframe.drop_duplicates(primary_key))

    if num_duplicated > 0:
        print(f'There are {num_duplicated} duplicated rows!')

    return num_duplicated > 0

####################################################################################################################################
# Function that treats the name of columns for Android Permission dataset:

def correct_col_name(name):
    """
    Function that treats the name of columns for Android Permission dataset.
    
    :param name: original name of the column.
    :type name: string.
    
    :return: cleaned name of the column.
    :rtype: string.
    """
    name = name.lower().replace('default : ', '').replace(' ', '_').replace('_:_','_')
    name = name.split('_(')[0].replace('-', '_').replace('/', '_').replace('.', '_').replace('\'', '')
    
    return name

####################################################################################################################################
# Function that returns labels from predictions as they are compared against a threshold:

def predict_label(score: float, threshold: float, labels: list) -> Union[str, int]:
    """
    Function that returns labels from predictions as they are compared against a threshold.

    :param score: predicted probability.
    :type score: float.
    :param threshold: value above which a probability leads to positive class.
    :type threshold: float.
    :param labels: labels for non-positive and positive classes.
    :type labels: list.

    :return: predicted label given predicted probability and threshold.
    :rtype: string or integer.
    """
    if len(labels) < 2:
        raise ValueError('Insufficient number of classes.')

    return labels[1] if score > threshold else labels[0]
