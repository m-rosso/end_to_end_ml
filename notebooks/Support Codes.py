#########################################################################################################################################

vars,feats=[],[]

for v in list(data_und[data_und.n_unique==2]['feature']):
  vars.append(df_train[v].var())
  feats.append(v)

pd.DataFrame(data={
    'feats': feats, 'vars': vars
}).sort_values('vars', ascending=False).head(50).reset_index(drop=True)



#########################################################################################################################################

check=[]
for p in df_train['package'].unique():
  check.append(np.nansum(df_train['related_apps'].apply(lambda x: np.NaN if pd.isna(x) else p in x)))

check=[]
for p in df_train['app'].unique():
  check.append(np.nansum(df_train['related_apps'].apply(lambda x: np.NaN if pd.isna(x) else p in x)))



#########################################################################################################################################

a=' '.join(list(df_train['description'].apply(lambda x: '' if pd.isna(x) else x.lower()))).split(' ')
len(set(a))



#########################################################################################################################################

df_train[df_train['num_known_apps'].isnull()][['related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(10)
df_train[df_train['num_known_malwares'].isnull()][['related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(10)
df_train[df_train['share_known_malwares'].isnull()][['related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(10)



#########################################################################################################################################

df_train.share_known_malwares.value_counts()

df_train[df_train['share_known_malwares']==0][['app_id', 'related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(3)
df_train[df_train.app_id==23600]['related_apps'].iloc[0]
df_train[df_train.package=='com.sillens.shapeupclub']

df_train[df_train['share_known_malwares']==0.5][['app_id', 'related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(3)
df_train[df_train.app_id==17573]['related_apps'].iloc[0]
df_train[df_train.package=='com.mshift.bankplus2go']

df_train[df_train['share_known_malwares']==1][['app_id', 'related_apps', 'num_known_apps', 'num_known_malwares', 'share_known_malwares']].sample(3)
df_train[df_train.app_id==23204]['related_apps'].iloc[0]
df_train[df_train.package=='fr.vdl.metroide']



#########################################################################################################################################

aa = df_train.dropna()
aa['check'] = aa[['package', 'related_apps']].apply(lambda x: x['package'] in x['related_apps'].split(', '),
                                                   axis=1)
aa[aa.check==True][['package', 'related_apps']]



#########################################################################################################################################

aa = df_train[['app', 'package', 'related_apps']].dropna()
aa['aa'] = [x+'.'+y for x,y in zip(aa['package'], aa['app'])]
aa['aa'] = aa.aa.apply(lambda x: x.replace(' ', ''))
aa['related_apps'].apply(lambda x: sum([1 if a in list(aa['aa']) else 0 for a in x.split(', ')])).describe()



#########################################################################################################################################

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
aaa=min_max_scaler.fit_transform(df_train[['L#share_known_malwares']])
display(pd.DataFrame(data={'L#share_known_malwares': [i[0] for i in list(aaa)]})['L#share_known_malwares'].describe())
pd.DataFrame(data={'L#share_known_malwares': [i[0] for i in list(aaa)]}).hist()

stats = min_max_stats(data=df_train, to_scale=['L#share_known_malwares'], scale=1)
aaa=min_max_scale(data=df_train, to_scale=['L#share_known_malwares'], stats=stats)
display(aaa['L#share_known_malwares'].describe())
aaa['L#share_known_malwares'].hist()

df_train['L#price'] = df_train['L#price'].apply(lambda x: x - df_train['L#price'].min())
df_train['L#price'] = df_train['L#price'].apply(lambda x: x/df_train['L#price'].max())
display(df_train['L#price'].describe())
df_train['L#price'].hist()



#########################################################################################################################################

if log_transform:
    # Variables that should not be log-transformed:
    not_log = [c for c in df_train.columns if c not in cont_vars]
    print('\033[1mTraining data:\033[0m')
    df_train = applying_log_transf(dataframe=df_train, not_log=not_log)

    print('\033[1mTest data:\033[0m')
    df_test = applying_log_transf(dataframe=df_test, not_log=not_log)
    print('\n')



#########################################################################################################################################

if which_scale in ['standard_scale', 'min_max_scale']:
  to_scale = [c for c in df_train.columns if ('L#' in c)]

if (which_scale=='standard_scale') & (scale_all==False):
    stats = standardize_stats(df_train, to_scale)
    df_train_scaled = standardize_data(df_train, stats)
    df_test_scaled = standardize_data(df_test, stats)

elif (which_scale=='min_max_scale') & (scale_all==False):
    stats = min_max_stats(data=df_train, to_scale=to_scale, scale=1)
    df_train_scaled = min_max_scale(data=df_train, stats=stats)
    df_test_scaled = min_max_scale(data=df_test, stats=stats)

else:
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    print('\033[1mNo transformation performed!\033[0m')



#########################################################################################################################################

####################################################################################################################################
# Function that calculates statistics for standard scaling numerical data:

def standardize_stats(training_data, to_stand):
    """
    Function that calculates statistics for standard scaling numerical data.
    
    :param training_data: data from which statistics should be calculated.
    :type training_data: dataframe.
    :param to_stand: names of variables that are going to be standard scaled.
    :type to_stand: list.
    
    :return: means and standard deviations in dictionaries whose keys are names of variables and values are the
    corresponding values of statistics.
    :rtype: dictionary.
    """
    means = dict(training_data[[c for c in training_data.columns if c in to_stand]].mean())
    stds = dict(training_data[[c for c in training_data.columns if c in to_stand]].std())
    
    return {'means': means, 'stds': stds}



#########################################################################################################################################

####################################################################################################################################
# Function for standard scaling numerical data:

def standardize_data(application_data, stats):
    """
    Function for standard scaling numerical data.
    
    :param application_data: data whose numerical variables should be standard scaled.
    :type application_data: dataframe.
    :param stats: means and standard deviations in dictionaries whose keys are names of variables and values are
    the corresponding values of statistics.
    :type stats: dictionary.
    
    :return: data with numerical variables standard scaled.
    :rtype: dataframe.
    """
    standardized_app = application_data.copy()
    
    for k in stats['means']:
        standardized_app[k] = standardized_app[k].apply(lambda x: (x-stats['means'][k])/stats['stds'][k])
        
    return standardized_app



#########################################################################################################################################

####################################################################################################################################
# Function that produces the statistics for min-max scaling numerical data:

def min_max_stats(data: pd.DataFrame, to_scale: list, scale=1):
    """
    Function that produces the statistics for min-max scaling numerical data.

    :param data: dataframe of reference for calculating the values used during min-max scaling.
    :type data: pandas dataframe.
    :param to_scale: names of columns that should be min-max scaled.
    :type to_scale: list.
    :param scale: scale of transformed data.
    :type scale: float or integer.

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



#########################################################################################################################################

####################################################################################################################################
# Function that applies the min-max scaling of numerical variables:

def min_max_scale(data: pd.DataFrame, stats: dict):
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



#########################################################################################################################################

check=df_train_scaled.drop(drop_vars, axis=1)==testando_som_train.drop(drop_vars, axis=1)
print(check.sum().sum())
print(np.prod(check.shape))

check=df_test_scaled.drop(drop_vars, axis=1)==testando_som_test.drop(drop_vars, axis=1)
print(check.sum().sum())
print(np.prod(check.shape))



#########################################################################################################################################

transf_data = applying_one_hot(df_train_scaled, cat_vars, test_data=df_test_scaled)
df_train_scaled = transf_data['training_data']
df_test_scaled = transf_data['test_data']

print(f'\033[1mShape of df_train_scaled:\033[0m {df_train_scaled.shape}.')
print(f'\033[1mShape of df_test_scaled:\033[0m {df_test_scaled.shape}.')



#########################################################################################################################################

treat_missings = TreatMissings(method='create_binary', drop_vars=drop_vars, cat_vars=cat_vars)
testando_som_train = treat_missings.fit_transform(df_train_scaled)

check = df_train_scaled.drop(drop_vars, axis=1)==testando_som_train.drop(drop_vars, axis=1)
check.sum().sum()==np.prod(check.shape)

treat_missings = TreatMissings(method='create_binary', drop_vars=drop_vars, cat_vars=cat_vars)
testando_som_test = treat_missings.fit_transform(df_test_scaled)

check = df_test_scaled.drop(drop_vars, axis=1)==testando_som_test.drop(drop_vars, axis=1)
check.sum().sum()==np.prod(check.shape)



#########################################################################################################################################

if scale_all==False:
    print('\033[1mTreating missing values of training data...\033[0m')
    df_train_scaled = treating_missings(dataframe=df_train_scaled, cat_vars=cat_vars,
                                        drop_vars=drop_vars)

    print('\033[1mTreating missing values of test data...\033[0m')
    df_test_scaled = treating_missings(dataframe=df_test_scaled, cat_vars=cat_vars,
                                        drop_vars=drop_vars)



#########################################################################################################################################

testando_som_train = df_train_scaled.copy()
testando_som_test = df_test_scaled.copy()

if scale_all==False:
    # Object for missing values treatment:
    treat_missings = TreatMissings(method=which_missings_treat, drop_vars=drop_vars, cat_vars=cat_vars)
    df_train_scaled = treat_missings.treat_missings(data=df_train_scaled)
    df_test_scaled = treat_missings.treat_missings(data=df_test_scaled)

if scale_all==False:
    # Object for missing values treatment:
    treat_missings = TreatMissings(method=which_missings_treat, drop_vars=drop_vars, cat_vars=cat_vars, statistic=missings_treat_stat)
    testando_som_train = treat_missings.treat_missings(data=testando_som_train, training_data=testando_som_train)
    testando_som_test = treat_missings.treat_missings(data=testando_som_test, training_data=testando_som_train)

# if scale_all==False:
#     # Object for missing values treatment:
#     treat_missings = TreatMissings(method='impute_stat', drop_vars=drop_vars, cat_vars=cat_vars, statistic=missings_treat_stat)
#     testando_som_train = treat_missings.treat_missings(data=testando_som_train, training_data=testando_som_train)
#     testando_som_test = treat_missings.treat_missings(data=testando_som_test, training_data=testando_som_train)

check = df_train_scaled.drop(drop_vars, axis=1)==testando_som_train.drop(drop_vars, axis=1)
check.sum().sum()==np.prod(check.shape)

check = df_test_scaled.drop(drop_vars, axis=1)==testando_som_test.drop(drop_vars, axis=1)
check.sum().sum()==np.prod(check.shape)



#########################################################################################################################################

v='L#rating'

q1 = np.quantile(df_train_scaled[v], q=0.25)
q3 = np.quantile(df_train_scaled[v], q=0.75)
display(q1 - k*(q3 - q1), q3 + k*(q3 - q1))

display(df_train_scaled[df_train_scaled[v] < q1 - 3*(q3-q1)].head(3)[['app_id', v]])

display(testando_som_train[['app_id', v]].loc[12])



#########################################################################################################################################

display(np.quantile(df_train_scaled['L#price'], q=0.975))

display(df_train_scaled[df_train_scaled['L#price'] > np.quantile(df_train_scaled['L#price'], q=0.975)].sample(3)[['app_id', 'L#price']])

display(testando_som_train[['app_id', 'L#price']].loc[8707])



#########################################################################################################################################

# Dataframe with only continuous variables:
cont_train_df = df_train_scaled[[f'L#{c}' for c in cont_vars]]

# # Variance selection:
# selection = FeaturesSelection(method='variance', threshold=0.1)
# selection.select_features(inputs=cont_train_df)
# selected_features = selection.selected_features

try:
  # Correlation selection:
  selection = FeaturesSelection(method='correlation', threshold=0.8)
  selection.select_features(inputs=cont_train_df)
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base na correlação!\033[0m')
except:
  print('Erro na seleção de variáveis com base na correlação!')

try:
  # Supervised learning selection:
  selection = FeaturesSelection(method='supervised', threshold=0)
  selection.select_features(inputs=df_train_scaled.drop(drop_vars, axis=1),
                            output=df_train_scaled['class'],
                            estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base na seleção supervisionada!\033[0m')
except:
  print('Erro na seleção supervisionada de variáveis!')

try:
  # Recursive feature elimination:
  selection = FeaturesSelection(method='rfe', num_folds=5, metric='roc_auc', step=5)
  selection.select_features(inputs=df_train_scaled.drop(drop_vars, axis=1),
                            output=df_train_scaled['class'],
                            estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base no método RFE!\033[0m')
except:
  print('Erro na seleção de variáveis via RFE!')

try:
  # Recursive feature elimination with cross-validation:
  selection = FeaturesSelection(method='rfecv', num_folds=5, metric='roc_auc',
                                min_num_feats=1, step=1)
  selection.select_features(inputs=df_train_scaled.drop(drop_vars, axis=1),
                            output=df_train_scaled['ViolentCrimesPerPop'],
                            estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base no método RFECV!\033[0m')
except:
  print('Erro na seleção de variáveis via RFECV!')

try:
  # Sequential feature selection:
  selection = FeaturesSelection(method='sequential', num_folds=5, metric='roc_auc', step=5, direction='forward')
  selection.select_features(inputs=df_train_scaled.drop(drop_vars, axis=1),
                            output=df_train_scaled['class'],
                            estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base no método SFS!\033[0m')
except:
  print('Erro na seleção de variáveis via SFS!')

try:
  # Random selection:
  selection = FeaturesSelection(method='random_selection', num_folds=5, metric='roc_auc',
                                max_num_feats=100, step=10)
  selection.select_features(inputs=df_train_scaled.drop(drop_vars, axis=1),
                            output=df_train_scaled['class'],
                            estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
  selected_features = selection.selected_features
  print(f'\033[1m{len(selected_features)} variáveis foram selecionadas com base na seleção aleatória!\033[0m')
except:
  print('Erro na seleção aleatória de variáveis!')



#########################################################################################################################################

method = 'correlation' # Defines which features selection method should be implemented. Choose among ['variance', 'correlation', 'supervised', 'rfe',
# 'rfecv', 'sequential', 'random_selection'].
threshold = 0.9 # Parameter of variance, correlation, and supervised learning selection.
num_folds = 5 # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).
metric = 'roc_auc' # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).
min_num_feats = 10 # Parameter of exaustive methods (RFECV).
max_num_feats = 80 # Parameter of exaustive methods (RFE, sequential selection, random selection).
step = 5 # Parameter of exaustive methods (RFE, RFECV, random selection).
direction = 'forward' # Parameter of exaustive methods (sequential selection).
regul_param = 1.0 # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).

# Dataframe with only continuous variables:
cont_train_df = df_train_scaled[[f'L#{c}' for c in cont_vars]]

# Features selection:
selection = FeaturesSelection(method=method, 
                                                            threshold=threshold,
                                                            num_folds=num_folds, metric=metric, min_num_feats=min_num_feats, max_num_feats=max_num_feats, step=step,
                                                            direction=direction)
selection.select_features(inputs=cont_train_df if method in ['variance', 'correlation'] else df_train_scaled.drop(drop_vars, axis=1),
                                                    outputs=df_train_scaled['class'],
                                                    estimator=LogisticRegression(penalty='l1', solver='liblinear', C=regul_param))
selected_features = selection.selected_features



#########################################################################################################################################

log_transform = True # Declare whether to log-transform numerical variables.
which_scale = 'standard_scale' # Declare which type of scaling should be applied over numerical variables ('standard_scale', 'min_max_scale', 'no_scale').
scale_all = False # Declare whether all variables (not only the continuous) are subject to scaling.
which_missings_treat = 'create_binary' # Declares which method of missing values treatment should be implemented ('create_binary', 'impute_stat').
missings_treat_stat = 'mean' # Declares which statistic should be used for missing values treatment ('mean', 'median').
cat_transf_var = 0.01 # Variance of dummy variables below which the respective categories are dropped out during categorical data transformation.
treat_outliers = True # Indicates whether outliers should be treated.
first_treat_outliers = False # Indicates whether outliers should be treated prior to all data transformations.
outliers_method = 'iqr' # Method for treating outliers.
quantile = 0.025 # Quantile parameter for outliers treatment.
k = 3 # Parameter for IQR outliers treatment.

method = 'correlation' # Defines which features selection method should be implemented. Choose among ['variance', 'correlation', 'supervised', 'rfe',
# 'rfecv', 'sequential', 'random_selection'].
threshold = 0.9 # Parameter of variance, correlation, and supervised learning selection.
num_folds = 5 # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).
metric = 'roc_auc' # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).
min_num_feats = 10 # Parameter of exaustive methods (RFECV).
max_num_feats = 80 # Parameter of exaustive methods (RFE, sequential selection, random selection).
step = 5 # Parameter of exaustive methods (RFE, RFECV, random selection).
direction = 'forward' # Parameter of exaustive methods (sequential selection).
regul_param = 1.0 # Parameter of exaustive methods (RFE, RFECV, sequential selection, random selection).



#########################################################################################################################################

solution_space_dim = ['scale_all', 'treat_outliers', 'outliers method', 'first_treat_outliers', 'method']
solution_space = []

# Loop over possible solutions:
for i in product(
  list_scale_all,
  list_treat_outliers,
  list_outliers_method,
  list_first_treat_outliers,
  list_method
):
  solution_space.append(i)

print(f'There are {len(solution_space)} possible solutions given the solution space dimensions {solution_space_dim}.')



#########################################################################################################################################

# Loop over trained models:
for m in models:
    print('-----------------------------------------------------------------------------------')
    print(f'\033[1m{m.replace("_", " ").capitalize()} model:\033[0m\n')
    display(predictions[m].groupby('y_true')[['test_score']].describe())

    plot_histogram(data=predictions[m], x=['test_score'], pos=[(1,1)], by_var=['y_true'],
                   barmode='overlay', opacity=0.75,
                   x_title=['y_hat'], y_title=['frequency'],
                   titles=['Distribution of predictions by true label'], width=600, height=450)
    
    plot_boxplot(data=predictions[m], x=['y_true'], y=['test_score'], pos=[(1,1)],
                titles=['Distribution of predictions by true label'], width=600, height=450)
    
    # Rate of y = 1 by decile of scores:
    predictions[m]['decile'] = pd.qcut(predictions[m]['test_score'], q=10)
    y_avg_dec = predictions[m].groupby('decile').mean()[['y_true']].reset_index()
    y_avg_dec['score'] = [str(d) for d in y_avg_dec['decile']]
    plot_bar(data=y_avg_dec, x=['score'], y=['y_true'], pos=[(1,1)],
             titles=['Rate of y = 1 by decile of scores'], width=600, height=450)
    print('-----------------------------------------------------------------------------------\n')



#########################################################################################################################################

[sum([v*w for v, w in zip(p, weights)]) for p in zip(*predictions)]



#########################################################################################################################################

inputs = X_test.iloc[0:2, :]

models_ = ['logistic_regression', 'logistic_regression']
predictions = []

# Loop over models:
for m in models_:
    if 'predict_proba' in dir(models[m].model):
        predictions.append(models[m].model.predict_proba(inputs))

    elif isinstance(models[m].model, lgb.Booster):
        predictions.append(models[m].model.predict(inputs))

    elif isinstance(models[m].model, xgb.Booster):
        xg_inputs = xgb.DMatrix(data=inputs)
        predictions.append(models[m].model.predict(xg_inputs))

    else:
        raise ValueError(f'Model {self.models.index()} is not from sklearn, LightGBM or XGBoost.')

print([x for x in zip(*predictions)])
print('\n')
weights = [1/2,1/2]
print([Ensemble.weighted_mean(p, weights) for p in zip(*predictions)])

inputs = X_test.iloc[0:2, :]

models_ = ['logistic_regression', 'light_gbm', 'xgboost']
predictions = []

# Loop over models:
for m in models_:
    if 'predict_proba' in dir(models[m].model):
        predictions.append(models[m].model.predict_proba(inputs)[:, 0])

    elif isinstance(models[m].model, lgb.Booster):
        predictions.append(models[m].model.predict(inputs))

    elif isinstance(models[m].model, xgb.Booster):
        xg_inputs = xgb.DMatrix(data=inputs)
        predictions.append(models[m].model.predict(xg_inputs))

    else:
        raise ValueError(f'Model {self.models.index()} is not from sklearn, LightGBM or XGBoost.')

print([x for x in zip(*predictions)])
weights = [1/3, 1/3, 1/3]
print([Ensemble.weighted_mean(p, weights) for p in zip(*predictions)])



#########################################################################################################################################

predictions = []

# Loop over models:
for model in models_:
    if 'predict_proba' in dir(model):
        predictions.append(model.predict_proba(inputs)[:, 0])

    elif isinstance(model, lgb.Booster):
        predictions.append(model.predict(inputs))

    elif isinstance(model, xgb.Booster):
        xg_inputs = xgb.DMatrix(data=inputs)
        predictions.append(model.predict(xg_inputs))

    else:
        raise ValueError(f'Model {self.models.index(model)} is not from sklearn, LightGBM or XGBoost.')

predictions



#########################################################################################################################################

inputs = X_test.iloc[0:100, :]
models_ = [models['logistic_regression'].model, models['light_gbm'].model, models['xgboost'].model]

ensemble = Ensemble(models=models_, statistic='weighted_mean', weights=[1/3, 1/3, 1/3], task='binary_class')
ensemble.predict(inputs=inputs, predict_class=False, threshold=0.5)

ensemble = Ensemble(models=models_, statistic='weighted_mean', weights=[1/3, 1/3, 1/3], task='binary_class')
ensemble.predict(inputs=inputs, predict_class=True, threshold=0.5)



#########################################################################################################################################

inputs = X_test
models_ = [models['logistic_regression'].model, models['light_gbm'].model, models['xgboost'].model]

ensemble = Ensemble(models=models_, statistic='weighted_mean', weights=[0, 0, 1], task='binary_class')
preds = ensemble.predict(inputs=inputs, predict_class=False)

preds = pd.DataFrame(data={
    'test_score_check': preds
})

pd.concat([preds, models['xgboost'].test_scores], axis=1, sort=False)



#########################################################################################################################################

[''.join(i) for i in zip(*[[f'X = {a}<br>' for a in [1,2]], [f'Y = {a}<br>' for a in [3,4]]])]



#########################################################################################################################################

to_log = [c for c in df_train.columns if c in cont_vars]
oper = LogTransformation(to_log=to_log)
testando_som = oper.fit_transform(data=df_train)

print(df_train['price'].iloc[1])
print(testando_som['L#price'].iloc[1])
print(np.log(1.41+0.0001))



#########################################################################################################################################

to_log = [c for c in df_train.columns if c in cont_vars]
oper = LogTransformation(to_log=to_log)
testando_som = oper.fit_transform(data=df_train)

to_scale = [f'L#{c}' for c in df_train.columns if c in cont_vars]
oper = ScaleNumericalVars(to_scale=to_scale, which_scale='standard_scale')
oper.fit(training_data=testando_som)
testando_som = oper.transform(data=testando_som)
display(testando_som[to_scale].describe())



#########################################################################################################################################

to_log = [c for c in df_train.columns if c in cont_vars]
oper = LogTransformation(to_log=to_log)
testando_som = oper.fit_transform(data=df_train)

to_scale = [f'L#{c}' for c in df_train.columns if c in cont_vars]
oper = ScaleNumericalVars(to_scale=to_scale, which_scale='min_max_scale')
oper.fit(training_data=testando_som)
testando_som = oper.transform(data=testando_som)
display(testando_som[to_scale].describe())



#########################################################################################################################################

which_missings_treat='create_binary'

oper = TreatMissings(vars_to_treat=vars_to_treat, method=which_missings_treat, drop_vars=drop_vars, cat_vars=cat_vars,
                     statistic=missings_treat_stat)
testando_som = oper.fit_transform(data=df_train, training_data=df_train)
print(testando_som['share_known_malwares'].isnull().sum())

print(df_train['share_known_malwares'].iloc[2])
print(testando_som['share_known_malwares'].iloc[2])
print(testando_som['NA#share_known_malwares'].iloc[2])

print(df_train['share_known_malwares'].iloc[3])
print(testando_som['share_known_malwares'].iloc[3])
print(testando_som['NA#share_known_malwares'].iloc[3])



#########################################################################################################################################

which_missings_treat='impute_stat'
missings_treat_stat = 'mean'
# missings_treat_stat = 'median'

oper = TreatMissings(vars_to_treat=vars_to_treat, method=which_missings_treat, drop_vars=drop_vars, cat_vars=cat_vars,
                     statistic=missings_treat_stat)
testando_som = oper.fit_transform(data=df_train, training_data=df_train)
print(testando_som['share_known_malwares'].isnull().sum())

print(df_train['share_known_malwares'].iloc[2])
print(testando_som['share_known_malwares'].iloc[2])
print(df_train['share_known_malwares'].mean())
# print(df_train['share_known_malwares'].median())

print(df_train['share_known_malwares'].iloc[3])
print(testando_som['share_known_malwares'].iloc[3])



#########################################################################################################################################

to_log = [c for c in df_train.columns if c in cont_vars]
oper = LogTransformation(to_log=to_log)
testando_som = oper.fit_transform(data=df_train)
print(df_train['price'].quantile(1-quantile), '\n')
print(df_train['price'].iloc[0])
print(df_train['price'].iloc[11], '\n')

oper = OutliersTreat(vars_to_treat=[c for c in cont_vars], method='quantile', quantile=quantile, k=k)
oper.fit(training_data=df_train)
testando_som = oper.transform(data=df_train)
print(testando_som['price'].iloc[0])
print(testando_som['price'].iloc[11])



#########################################################################################################################################

pipeline = Pipeline(
    operations = [
                  LogTransformation(to_log=to_log),
                  ScaleNumericalVars(to_scale=to_scale, which_scale=which_scale),
                  TreatMissings(vars_to_treat=vars_to_treat, method=which_missings_treat, drop_vars=drop_vars, cat_vars=cat_vars,
                                statistic=missings_treat_stat),
                  OneHotEncoding(categorical_features=cat_vars, variance_param=cat_transf_var)
    ]
)

_, testando_som1 = pipeline.transform(data_list=[df_test], training_data=df_train)
testando_som1 = testando_som1[0]

if scale_all==False:
    # Assessing missing values (training data):
    missings_detection(df_train_scaled.drop([v for v in drop_vars if v!='class'], axis=1), name=f'df_train_scaled')

    # Assessing missing values (test data):
    missings_detection(df_test_scaled.drop([v for v in drop_vars if v!='class'], axis=1), name=f'df_test_scaled')

    # Checking datasets structure:
    testando_som1 = data_consistency(dataframe=df_train_scaled, test_data=testando_som1)['test_data']

check = df_test_scaled.drop(drop_vars, axis=1)==testando_som1.drop(drop_vars, axis=1)
if np.prod(df_test_scaled.drop(drop_vars, axis=1).shape)!=check.sum().sum():
    print('Revisar o código criado!')



#########################################################################################################################################

from utils import correct_col_name

# 
schema = dict(
    zip(
        [c for c in input_data.drop(['class'], axis=1).columns],
        ['str' if type(input_data[c].iloc[0])==str else 'numeric' for c in input_data.drop(['class'], axis=1).columns]
    )
)

for sample in [('Dirty Jokes', 'com.appspot.swisscodemonkeys.dirty', 0), ("Trans'Alerte", "fr.trans.alerte", 10)]:
    input_data = pd.read_csv('../data/Android_Permission.csv')

    # Columns names:
    input_data.columns = [correct_col_name(c) for c in input_data.columns]

    print(f'Shape of input_data: {input_data.shape}.')

    # Removing duplicates:
    input_data.drop_duplicates(inplace=True)
    print(f'Number of instances after removing duplicates: {len(input_data)}.')

    input_data = input_data[(input_data.app==sample[0]) & (input_data.package==sample[1])]

    # 
    model = Model(schema=schema, pipeline=pipeline, ensemble=ensemble, variables=variables)

    # 
    prediction = model.predict(input_data=dict(zip(input_data.iloc[0].index, input_data.iloc[0].values)),
                               training_data=df_train)

    if prediction != ensemble.predict(X_test.iloc[sample[2]]):
        print('Erro no cálculo do score!')



#########################################################################################################################################

from utils import correct_col_name

input_data = pd.read_csv('../data/Android_Permission.csv')

# Columns names:
input_data.columns = [correct_col_name(c) for c in input_data.columns]

print(f'Shape of input_data: {input_data.shape}.')

# Removing duplicates:
input_data.drop_duplicates(inplace=True)
print(f'Number of instances after removing duplicates: {len(input_data)}.')

input_data = input_data[(input_data.app=='Dirty Jokes') & (input_data.package=='com.appspot.swisscodemonkeys.dirty')]

# 
schema = dict(
    zip(
        [c for c in input_data.drop(['class'], axis=1).columns],
        ['str' if type(input_data[c].iloc[0])==str else 'numeric' for c in input_data.drop(['class'], axis=1).columns]
    )
)

models_ = [models[m].model for m in ['light_gbm']]
weights = [1/(len(models_)) for i in range(len(models_))]
ensemble = Ensemble(models=models_, statistic='weighted_mean', weights=weights, task='binary_class')
### Substituir essa declaração pela importação dos objetos "pipeline" e "ensemble".

# 
model = Model(schema=schema, pipeline=pipeline, ensemble=ensemble, variables=variables)

# 
prediction = model.predict(input_data=dict(zip(input_data.iloc[0].index, input_data.iloc[0].values)),
                           training_data=df_train)
prediction

ensemble.predict(X_test.iloc[0])



#########################################################################################################################################

def bacon_com_ovos(n):
    assert isinstance(n, int), 'n deve ser um inteiro'

    if (n % 3 == 0) and (n % 5 == 0):
        return "Bacon com ovos"

    if n % 3 == 0:
        return "Bacon"

    if n % 5 == 0:
        return "Ovos"

    return "Fome"

import unittest

class TestBaconComOvos(unittest.TestCase):
    def test_bacon_com_ovos_assert_error_caso_sem_int(self):
        with self.assertRaises(AssertionError):
            bacon_com_ovos('0')

    def test_bacon_com_ovos_baconcomovos_caso_mult3e5(self):
        entradas = (15, 30, 45, 60)
        saida = 'Bacon com ovos'

        for entrada in entradas:
            with self.subTest(entrada=entrada, saida=saida):
                self.assertEqual(bacon_com_ovos(entrada), saida, msg=f'{entrada} nao retornou {saida}')

    def test_bacon_com_ovos_fome_caso_nao_mult3e5(self):
        entradas = (1, 2, 4, 7)
        saida = 'Fome'

        for entrada in entradas:
            with self.subTest(entrada=entrada, saida=saida):
                self.assertEqual(bacon_com_ovos(entrada), saida, msg=f'{entrada} nao retornou {saida}')

    def test_bacon_com_ovos_bacon_caso_mult3(self):
        entradas = (3, 6, 9, 12)
        saida = 'Bacon'

        for entrada in entradas:
            with self.subTest(entrada=entrada, saida=saida):
                self.assertEqual(bacon_com_ovos(entrada), saida, msg=f'{entrada} nao retornou {saida}')

    def test_bacon_com_ovos_ovos_caso_mult5(self):
        entradas = (5, 10, 20, 25)
        saida = 'Ovos'

        for entrada in entradas:
            with self.subTest(entrada=entrada, saida=saida):
                self.assertEqual(bacon_com_ovos(entrada), saida, msg=f'{entrada} nao retornou {saida}')

unittest.main(argv=[''], verbosity=2, exit=False)



#########################################################################################################################################

A predictive model consists of a function that converts input variables into a prediction, and there are lots of different learning algorithms which use pairs of input variables and predefined labels from a training dataset to estimate the collection of parameters that constitute the predictive function. Given the existing diversity of learning methods, it is possible to generate predictors from an **ensemble of models** that combines predictions of different single models. Consequently, a predictive model can be either simple or composite, where the first type is defined by a single model and the second by a collection of models.



#########################################################################################################################################

