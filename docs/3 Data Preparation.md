## Data preparation
In this stage, Python modules were developed with the objectives of simplifying and customizing the implementation of *data preparation* tasks. Besides, codes for using such functions and classes were created and later used during experimentation in data modeling stage.

Data preparation implies in consistent and ready-to-use datasets for training, validating, and testing machine learning models during the data modeling stage. Activities of data preparation can be summarized as follows:
* Features classification and early selection.
* Data pre-processing (transformations and dataset composition).
* Features selection.

In sections below, we find the description of the *goals* of each data preparation task, the definition of which *functions and classes* were created, and the documentation of *assumptions* made for implementing them (operations order, values of parameters, and so on).

--------------
### Features classification and early selection
Activities of this task help understanding data about how each variable should be processed. Therefore, **features classification** indicates whether a given input variable is *numerical, binary or categorical*, so the appropriate transformation can be applied over it. The **early selection** of variables simply selects which among the complete collection of input variables have a *sufficient variance in the training data* and a *share of missing data smaller than a predefind threshold*.

A function named *classify_variables*, available in Python module *utils.py* of *src* folder, implements both the features classification and the early selection by taking as arguments the training data, flags indicating whether to drop variables with just a few variance or with excessive number of missings, and the respective thresholds for the variance and share of missings.

The *classify_variables* function, as used in this project, first drops all variables with a share of missings in the training data higher than 95%, then classifies them into numerical, binary or categorical, and finally drops those numerical variables with zero variance in training data.

Finally, another selection of variables implemented here drops **binary variables based on their variance**. More specifically, binary variables with variance in the training data smaller than 0.01 were dropped. Note that additional empirical criteria could be raised to further select variables in this stage.

--------------
### Data pre-processing
These activities are the core task of data preparation stage, since they implement operations that make training data ready to feed machine learning models and to benefit the fitting so their model performances get improved. **Note that all operations have different ways of being implemented, thus leading to several distinct pipelines to be tested during model experimentation.**

#### Transformations
Here, information present in a given input variable is transformed in such a way that it can be better used by machine learning algorithms. Depeding on the data type of a variable, different transformations can be applied. Below are listed transformations regarding **numerical variables**:
* **Logarithmic transformation:** consists of applying the logarithm function over non-negative values of a variable (plus a constant for handling zero values). This transformation helps to adjust the scale of numerical variables and to prevent outliers from distorting their distribution.
    * Class *LogTransformation*, available in *transformations.py* module in *src*, were developed, requiring only the data to fit and transform (through method *fit_transform*) and the list of columns names of those variables that should be log-transformed. This class returns the value 0 for original negative values.
* **Scaling transformation:** two different methods for bringing all numerical variables to the same scale were made available for tests: *standard scaling*, that makes all variables have zero mean and unity standard deviation, and *min-max scale*, which makes values of all variables to range between 0 and 1. Both methods prevent that variables with different scale have their estimated effects on the target being contaminated by their magnitudes. *Scaling transformation should be used after logarithmic transformation because scaling may lead to negative values.*
    * Class *ScaleNumericalVars*, available in *transformations.py* module in *src*, were developed to implement both standard scaling and min-max scaling. This class should be initialized by indicating which method to used, after which training data is passed to *fit* method and application data is given to the *transform* method.
* **Missing values treatment:** missing values of numerical variables not only can not be used for fitting many machine learning algorithms but can also provide useful information for prediction. Consequently, two methods were considered here: imputation of statistics (mean or median) and creation of a binary variable indicating whether an observation has missing for a given variable and then the imputation of 0 for that variable. *Missing values treatment was implemented after logarithmic and scaling transformations because scaling requires the calculation of statistics that could be distorted by the imputation of values.*
    * Class *TreatMissings*, available in *transformations.py* module in *src*, were developed considering both methods above mentioned. It requires parameters regarding the chosen method of treatment during initialization, training data when applying *fit* method (which calculates statistics when needed) and application data for *transform* method.     
        * *Note that this class also imputes "missing_value" for textual variables, thus creating a missing values category for them.*

Additionally, some further transformations that could improve model fitting are polynomial functions and interaction among variables (irrespective of their data types).

Only one transformation was used regarding **categorical data**, since no other ordered variables than the rating of apps (treated here as numerical) were available.
* **One-hot encoding:** converts each category of a variable into a binary variable that assumes 1 whether an observation belongs to that category and 0 otherwise.
    * Class *OneHotEncoding*, available in *transformations.py* module in *src*, were developed. When fitting the data, this class treats textual variables using regex methods and drops those binary variables whose variance is smaller than some predefined threshold.

#### Dataset composition
This section covers all operations that process the **dataset as a whole**, focusing either on rows or columns that compose it. The main difference between these activities and those of transformations is methodological. Here, operations are performed having the dataset overall description in mind, so *outliers* or anomalous observations are identified, the *balance* (distribution) between target labels are assessed and eventually adjusted, observations receive different *weights*, observations are *sampled* for reducing the dataset size, or the label values of target variable are redefined (*labeling*).

Even so, only outliers treament was considered in this project by using the *OutliersTreat* class that implement two different methods: identification of outliers based on either a given quantile of training data or the interquartile range (IQR), while in both cases outlier values are replaced by the quantile or the IQR value (lower or upper bound), respectively. *Outliers treatment can take place either before any numerical data transformation or after all those operations were applied, and both alternatives were considered during model experimentation.*

Finally, a class named *Pipeline* were created to reproduce a given data pipeline implemented using all the above mentioned Python classes that were created during this project. After fitting and transforming data with an object of this class, training and testing datasets are created.

--------------
### Features selection
This final stage of data preparation takes data which is ready to be used for training models and applies supervised learning methods (i.e., methods that use the target variable) for selecting those variables that have the highest expected predictive power. Here, the following methods were considered:
* No features selection.
* scikit-learn methods of *recursive feature elimination (RFE)*: RFE and RFECV.
* Training a simple and regularized model to identify those variables whose contribution to the fitted model is null.
**Again, different features selection methods and their distinct ways of being used lead to multiple alternatives that can be tested during model experimentation.**

For implementing features selection in this project, the customized class *FeaturesSelection* of *features_selection.py* module was used.
