## Data modeling
This stage is the core of any machine learning project, since it is here that a predictive, or more generally, **an analytical solution is developed** making use of assumptions defined during discussion stage and the data formatted and prepared during data engineering and preparation stages.

Data modeling involves multiple sets of related activities, but before listing them it is important to make some definitions explicit. First, a **data pipeline** consists of all operations of data transformation and dataset composition (*pipeline of data preparation*). Second, a **model** refers to all estimated parameters that configure a predictor and that are derived from a learning algorithm. Besides, a model can be either simple or composite, i.e., made of a single model or an **ensemble** of models, respectively. A **complete data pipeline**, in its turn, combines a data pipeline followed by a fitted predictive model.

Therefore, a complete data pipeline is able to take a raw input data point and return a predicted value of the response variable, here, the probability of an app being a malware. The development of a complete data pipeline involves sequentially performing the following activities:
* Data modeling starts by defining which **learning algorithms** will be tested, together with which **hyper-parameters** will be optimized.
* Then, **experimentation** takes place by iterating different configurations of a complete data pipeline.
* **Model evaluation** procedures are applied over all developed complete pipelines, which involves defining multiple **ensembles of models**.
* Next, the **best complete pipeline** is selected and then a new round of experimentation is implemented by **fine tuning** it.
* Finally, **artifacts for deployment** are created, ending the data modeling stage.

Below, each of these sets of activities are described in more detail as they were executed in this project.

--------------
### Initial selection of learning algorithms
Since *explicability* and *interpretability* are not priority in this project, which focus on maximizing model accuracy instead, a diverse collection of learning algorithms were tested during model experimentation:
* **Logistic regression of *sklearn***.
    * *Hyper-parameters under optimization:* regularization parameter $C$.
    * *Fixed hyper-parameters:* L1 regularization.
* **Random forest of *sklearn***.
    * *Hyper-parameters under optimization:* number of trees ($n\_estimators$).
    * *Fixed hyper-parameters:* maximum number of features that can be selected for splits ($max\_features$) given by the squared root of the number of inputs, minimum observations for split ($min\_samples\_split$) equal to 2.
* **Gradient boosting of *LightGBM***.
    * *Hyper-parameters under optimization:* fraction of observations randomly picked to construct a tree ($bagging\_fraction$), learning rate ($learning\_rate$), maximum depth of trees ($max\_depth$), number of trees ($num\_iterations$).
* **Gradient boosting of *XGBoost***.
    * *Hyper-parameters under optimization:* fraction of observations randomly picked to construct a tree ($subsample$), learning rate ($eta$), maximum depth of trees ($max\_depth$), number of trees ($num\_boost\_round$).
* **SVM of *sklearn***.
    * *Hyper-parameters under optimization:* degree of polynomial extension ($degree$).
    * *Fixed hyper-parameters:* regularization parameter ($C$) of 1, polynomial kernel ($kernel$), $scale$ equals to "gamma".

Check the notebook "Data Modeling - Experiments" for understanding the codes regarding the construction of models using these learning algorithms.

--------------
### Population definition
Even that different models could be developed for different categories of apps, all data was gathered and modeled together.

--------------
### Experimentation
This stage of data modeling defines, at each of its iterations, the **best model (either an ensemble of models or a single model) for a given pipeline of data preparation**. Consequently, the main user-defined configuration of the notebook "Data Modeling - Experiments" refers to data preparation activities, which should be modified at each iteration of experimentation. Namely, these are the parameters to be chosen in a given experiment:
* Whether to **scale all variables** or only the numerical ones, i.e., only those whose data type is originally numeric.
* Whether to **treat outliers**, and which **outliers method** to use (IQR or percentiles for defining if a variable value is an outlier).
* Whether to **first treat outliers** prior to any data transformation, or to treat them after applying transformations such as scaling the data.
* **Features selection method**: no selection, selection based on correlation among inputs, RFECV, or supervised selection (dropping inputs with null coefficient in a simple linear model).

Since the space solution over these 5 parameters amounts to 40 distinct pipelines, **sequential tests were implemented instead of a greedy exploration**. For instance, when the best solution to whether to scale all variables or not has been defined, the outliers treatment setting was defined, and so forth. If some evolutionary or bayesian optimization algorithm had been implemented, a better solution could be found, since more candidate solutions would be available as multiple combinations of those parameters would be tested.

While those data preparation arguments were under optimization during experimentation, other arguments were kept constant, such as the use of log-transformation and standard scaling of numerical variables, treatment of missing values by creating a binary variable that indicates whether an original input has missing value for a given observation, and drop of binary variables with variance smaller than 0.01 in training data. Additionally, in a given experiment, all learning algorithms previously chosen were trained by optimizing their hyper-parameters using K-folds CV.

Check the notebook "Data Modeling - Experiments" for more on how experimentation was implemented in this project.

--------------
### Model evaluation
Fitted models are evaluated for two main reasons: **validation** during the selection of hyper-parameters and for **testing** their predictive power when applying them over unseen data points, which reproduces the application of the model in a production environment.

The **validation strategy** assumed here makes use of *K-folds cross-validation* with 5 folds of non-shuffled data together with *grid/random search* that tries to maximize the average *ROC-AUC* calculated over the 5 folds of data. The **train-test strategy** for complete pipeline selection shuffles the available data and assignes 33% of the apps to the test dataset. As a result, after data engineering stage there are a training dataset with 18298 apps and a test dataset with 9012 apps, both of them with 191 columns.

**Performance metrics** that were calculated on the test dataset are the following: [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html), [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision), [MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html), [precision rate](https://en.wikipedia.org/wiki/Precision_and_recall#Precision), [recall rate](https://en.wikipedia.org/wiki/Precision_and_recall#Recall), [false negative rate](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates) and [false positive rate](https://en.wikipedia.org/wiki/False_positive_rate). As discussed in the document about the *Discussion stage*, it was not a priority in the development of this project, which also implies that no **operational/business metric** was calculated.

Check the notebook "Data Modeling - Experiments" for understanding the codes regarding the evaluation of constructed models.

Another crucial component of model evaluation is analysis of **features importances**. Estimated coefficients of logistic regression and the standard feature importance calculation of gradient boosting models (for *LightGBM* and *XGBoost*) are displayed for the final trained model in the notebook "Data Modeling - Model Analysis". This notebook also presents the **analysis of predictions** by calculating statistics regarding the following:
* Unconditional distribution of predicted probabilities of malware.
* Distribution of predicted probabilities of malware conditioned on the true label.
* Distribution of predicted probabilities of malware conditioned on input variables.

--------------
### Ensemble definition
Given a data pipeline of data preparation and once a model has been trained for each learning algorithm used here, different **ensembles** are constructed combining subsets of those individual models. These ensembles are then treated as combined models and gathered with all single models for model selection (check the next section for more).

Again for a given data pipeline of data preparation, only a **subset of the model space** was considered while creating ensembles because, from all combinations of fitted models taking any possible real-valued vector of weights, there were created only the ensemble of all fitted models with equal weights and the ensembles of the 2, 3 and 4 best models (i.e., highest test data ROC-AUC) also with equal weights.

In order to create ensembles of models, it was developed a Python class named *Ensemble*, which takes as initialization arguments a list of model objects, a string with the name of statistic to be computed over each prediction constructed from all individual models, such as the weighted mean of predictions (used in this project), and a list of weights for aggregating the predictions of individual models. This class has a *predict* method that returns, for each provided input data point, a prediction that is calculated as a function of the predictions created by all individual models that were declared during initialization.

Check the notebook "Data Modeling - Experiments" and the *production.py* module in *src* folder for more about the creation of model ensembles in this project.

--------------
### Complete pipeline selection
During the experimentation stage of data modeling two types of model selection were implemented.
* First, for a given iteration of experimentation, which implies in a given pipeline of data preparation, different machine learning algorithms were fitted and, from combinations of them, different ensembles were created.
* Then, for a given pipeline of data preparation, a **model selection** was done in order to define the best model or ensemble of models. This selection follows from finding the model/ensemble which maximizes the most of three performance metrics: ROC-AUC, MCC and accuracy. Thus, between two models it would be selected that with the highest values of at least 2 of these performance metrics.
* After defining the best model/ensemble for each of the tested pipelines of data preparation, a **complete pipeline selection** was done to choose, among all combinations of data preparation pipeline and model/ensemble, which one that maximizes the most of the same three performance metrics: ROC-AUC, MCC and accuracy..
* Consequently, the complete data pipeline selected is the best possible model as evaluated on different metrics calculated using the test data.
* Note that it is not necessary to compare all complete pipelines, since to every pipeline of data preparation there is one model/ensemble that is the best for it. Then, it is necessary to compare just among the best models/ensembles to find the best overall combination of data preparation pipeline and predictive model.

Check the notebooks "Data Modeling - Experiments" and "Data Modeling - Final Training" for understanding the codes regarding the evaluation of constructed models.

--------------
### Final model training
After finding the best complete data pipeline during model experimentation, notebook "Data Modeling - Final Training" implements a **fine tuning** of parameters that were kept constant so far. The following parameters were further optimized in this final activity of model training:
* **Method for scaling numerical variables:** no scaling, standard scaling and min-max scaling.
* **Method of missing values imputation:** creation of binary variables indicating missings, imputation of mean or median values.
* **Exclusion of some input variables**.

In addition to these changes in data pipeline, **hyper-parameters** were again optimized, given that the best model consists of a single LightGBM and its parameters were first defined using a random search that extracted 10 random combinations of hyper-parameters values.

Check the notebook "Data Modeling - Model Analysis" to understand all components of the complete data pipeline that was selected by the end of the data modeling stage.

--------------
### Explainability and interpretability
This very crucial stage of data modeling was not implemented yet, but it was inserted as top priority in the backlog.

--------------
### Artifacts for deployment
Ends the data modeling stage the creation of all **artifacts** needed for implementing the analytics solution that has the fitted data pipeline as its cornerstone. Notebook "Data Modeling - Final Training", after fine tuning the best complete data pipeline found during experimentation, exports to the *artifacts* folder the following files:
* **Training data:** used during feature engineering after receiving an input data point as well as to perform some transformations during the execution of the pipeline of data preparation.
* **List with names of input variables:** used to select only relevant variables after preparing the input data point, i.e., only those actually needed to generate predictions from the fitted (ensemble) model.
* Picke files with the **pipeline of data preparation** and the **ensemble of fitted models**: both are combined to create the object that consists of the complete data pipeline which takes a raw input data point, prepares it by cleaning the data, creating features and applying the pipeline of data transformations, and generates a prediction from the prepared data vector by using the ensemble of models.

The object with the complete data pipeline derives from a developed class named *Model*, present in module *production.py*, that is initialized by declaring the pipeline of data preparation, the ensemble of fitted models, and the list with names of input variables. Another initialization argument is a dictionary with the *schema* of expected raw input data points, whose keys are names of raw variables and values are their respective data types. The *predict* method of class *Model* takes a raw input data point as a dictionary whose keys are raw variables names and values are their respective values together with the training data so it can return a prediction of the target variable for the received observation.

--------------
### Additional backtests
Not implemented in this project.

--------------
### Tests in production
Not implemented in this project.
