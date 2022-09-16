## Data engineering
Data engineering can be seen here as a collection of ETL (extract, transform and load) operations, except for data understanding and EDA tasks, which have an analytical perspective. By the end of this chapter, data is ready to be processed in the appropriate way so the designed solution can be developed. The following activities are implemented in the notebook of *Data Engineering*:
* **Data understanding and cleaning:** definition of data types and domain of features, assessment of missing values, classification of features.
* **Exploratory data analysis:** unconditional and conditional distribution of features, distribution of target given features.
* **Feature engineering:** further creation of variables from original data.

--------------
### Data understanding and cleaning
Data understanding begins by **importing the data** and checking if there are any duplicated rows, or missings in the target variable or the primary key of the dataset.

Then, variables are classified according to their **data types**: object (i.e., text), integer or float. The **domain of variables** is also defined by getting the quantity of unique values for each variable together with samples of values from the data.

**Missing values assessment** indicates the number and share of missings by variable across observations. Additionally, the number of columns with missings by observation is also defined. In order to help with feature engineering and data preparation, the distribution of missings by label of target variable is calculated. All this activities ultimately help describing the collection of available features.

--------------
### Exploratory data analysis
Given the objectives of this project, the **EDA section was not a priority here**, reason why only some main statistics were calculated to give understanding about which variables are the most relevant in their relationship with the target label.

Even so, it is crucial to always run statistical tests in order to find the significance of results provided by descriptive statistics. Additionally, data visualization is also necessary when exploring the available data, specially because of different datasets can lead to the same set of basic statistics (see the [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) for more). Consequently, these activities were inserted into the **project's backlog**.

To see the results of exploratory data analysis in detail, please check the notebook *Data Engineering* (in folder *notebooks*), while here we find only their highlights. Results follow from statistics regarding the following **distributions calculated using the training dataset**: distribution of features ($P(X)$), distribution of target variable ($P(Y)$), distribution of features conditional on true label ($P(X|Y)$), and distribution of true label conditional on features ($P(Y|X)$).
* Majority of apps in the training data are malwares (67%).
* Apps receive one among 30 categories and the most frequent are: "Entertainment", "Travel & Local", "Books & Reference", "Arcade & Action", "Brain & Puzzle", "Casual" and "Personalization", ranging from 10% to 6% of training data.
* Most apps have no cost of download, and only malwares have a positive price.
* Malwares are only slightly worse evaluated (average of 3.97 vs. 3.26 of safe apps).
* The most risky apps are from the following categories: "Transportation", "Medical", "Travel & Local" and "Sports", which have a malware rate ranging from 99% to 96%.
* When running bivariate models (malware label against individual inputs), it can be found that number of ratings and price are the most relevant variables to individually predict the malware variable.

--------------
### Feature engineering
There were performed feature engineering operations over **textual variables** that indicate the name of apps that are related to each app in the dataset (*related apps*) and that describe each app (*description*).
* **Number of related apps:** intuitively, the more apps that relate to a given app the less likely it would be that it is a malware.
    * This feature is simply defined by counting the *number of related apps as indicated by the dataset*.
    * The distribution of this variable is similar for malwares and for safe apps.
* **Number of words in description:** one can think that legitimate apps have a longer description provided by their developers.
    * This feature counts the *number of words in the description variable without removing stop words*.
    * Opposite to what was expected, malwares have a somewhat larger number of words in their description.
* **Number and share of known related apps:** the more related apps that are also present in the historical data (*known apps*) the better is the information regarding the app, so it is less likely that it is a malware.
    * Using the training data as a reference to the historical data of apps, *each related app of an app is assessed whether it is known or not*.
    * Then, the *number of known related apps* is calculated and also the *share of known related apps among the complete collection of related apps*.
    * Safe apps are expected to have a larger number and share of known related apps.
* **Number and share of known related apps that are malwares:** as defined below, the more information about related apps of an app the better. The most important information is precisely that indicating if a related app is a malware. The more malwares that are related with an app, the higher the risk that it is a malware itself.
    * The existence of *related apps* in the training data is checked. Again, the training data is taken as a historical dataset of apps.
    * It is defined the number of these related apps in the training data that are *malwares*.
    * Finally, it is calculated the *number and share of malwares* in the collection of related apps of an app.
    * As expected, the more related apps that are malwares, the higher the risk that the app is also a malware.
    * Notes:
        * Even though the training data is being used here as the reference for finding malware related apps, a pre-training or an extension of the training data could be used for the feature construction.
        * If no related app is found in the training data, then the variable should receive a missing value (which is done by construction, since the denominator of the share is zero in such circumstance).
* **Natural language processing (NLP) over the description variable:** such feature engineering operation is a priority in the backlog. Some possibilities involve bag-of-words and TF-IDF, or statistical methods (such as log-likelihood ratio, LLR) for selecting risky words.
