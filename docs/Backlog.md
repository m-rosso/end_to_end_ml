## Backlog
This document gathers all activities that the project has not covered yet, as a matter either of time or scope. Inserting such activities here gives a broader picture of elements that can enrich the project, by providing more sophistication and complexity to the final product/service or by improving model performance and robustness.

The backlog is structured following sections of a standard script for machine learning development, very similar to CRISP-DM methodology. So, the following sections organize the collection of backlog activities: [discussion](#discussion)<a href='#discussion'></a>, [data engineering](#data_eng)<a href='#data_eng'></a>, [data preparation](#data_prep)<a href='#data_prep'></a>, [data modeling](#data_modeling)<a href='#data_modeling'></a>, [deployment](#deploy)<a href='#deploy'></a> and [documentation](#doc)<a href='#doc'></a>.

After each block of related activities, they are tagged according with two criteria:
* *Data analysis* or *data product*: in a data science project, a given activity can serve a limited scope, by answering questions specific to the current project (**data analysis**), or a broader purpose, by resulting in codes that can be used in any related project or task (**data product**).
* *Priority:* each activity receives a rate of priority (**1 to 5**, less to more relevant) for either improving the project or generating useful knowledge.

--------------
### Discussion<a id='discussion'></a>
First of all, the business problem should be understood, in addition to domain knowledge and product/service specification. Tasks are defined and scheduled.

1. Stages, tasks and scheduling.

* Make use of AWS development and deployment tools: CodeCommit + CodeBuild + CodeDeploy, or using CodePipeline (and CodeStar).
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 1*</ins>.

2. Terms and definitions.

* Glossary of terms and definitions adopted in the project.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 1*</ins>.

### Data engineering<a id='data_eng'></a>
Stage of data imports, understanding and cleaning. Besides, insights are developed through explorartory data analysis (EDA) and additional features are created using relatively complex operations.

1. Exploratory data analysis.

* Statistical tests and data visualization in the exploratory data analysis (EDA) stage.
	* <ins>*Data analysis backlog*</ins>.
	* <ins>*Priority: 1*</ins>.
	* Visualization for the distribution of input variables (unconditional and conditional on the outcome variable).
		* Histograms for continuous variables, barplots for categorical variables.
	* Visualization for the distribution of the outcome variable conditional on input variables.
		* Barplot of positive class rate for different levels of an input variable.
	* Tests for the difference in the mean of input variables for positive (y = 1) and non-positive classes (y = 0).
	* Tests for the difference in the mean of the outcome variable for different values of inputs.

2. Feature engineering.

* NLP for feature engineering based upon the "description" variable.
	* <ins>*Data analysis backlog*</ins>.
	* <ins>*Priority: 5*</ins>.

### Data preparation<a id='data_prep'></a>
Here, data is prepared for feeding analytical or predictive models. Data is adequately transformed and datasets for training, validation and test are composed.

1. Data pre-processing.

* Apply different **transformations**.
	 * Logarithmic transformation applied over a wider set of variables.
	 * Logarithmic transformation applied over a set of truly continuous variables (i.e., those having more than K > 2 unique values).
		* <ins>*Data analysis backlog*</ins>.
		* <ins>*Priority: 1*</ins>.
	* Additional methods for scaling numerical data: research and improvements over the "ScaleNumericalVars" class, test in pipelines.
	* Additional methods of missing values treatment: research and improvements over the "TreatMissings" class, test in pipelines.
		* <ins>*Data analysis and data product backlog*</ins>.
		* <ins>*Priority: 2*</ins>.
	* Polynomial functions of numerical variables and interaction among variables: raise possibilities of application in this project and development of Python classes for these transformations.
		* <ins>*Data analysis and data product backlog*</ins>.
		* <ins>*Priority: 3*</ins>.
* Formal tests of data transformation classes.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
* Apply techniques of **dataset composition**.
	* Additional methods for treating outliers: research and improvements over the "OutliersTreat" class, test in pipelines.
		* <ins>*Data analysis and data product backlog*</ins>.
		* <ins>*Priority: 4*</ins>.
* Reflect and discuss about how adequate are the data preparation procedures taken here, in general and for the dataset of this project.
* Review and discuss the restrictions of pipelines tested in this project.
	* <ins>*Data analysis backlog*</ins>.
	* <ins>*Priority: 4*</ins>.

### Data modeling<a id='data_modeling'></a>
Experiments are conducted in order to develop the best model possible. Then, a final model is trained and their outcomes are analyzed.

1. Experimentation: hyper-parameters tuning and model training.

* Script for model experimentation (constructed upon the Jupyter notebook).
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 2*</ins>.
	* This script should allow different specifications of data pipeline: implementations of data transformation, order of transformations, and so on.
* Formal tests of "Pipeline" and "Ensemble" classes.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
* Modify the class "Pipeline" in such a way that the training data need only to be provided when a "fit" method is run. Consequently, no training data would be necessary when using a fitted object of this class (i.e., when applying the "transform" method).
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
* Include in the class "Model" the possibility of transforming data so all variables are scaled.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
* Improve the method "__check_schema" of class "Model" so it checks if inputs respect the expected domain of attributes.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
* Development and application of optimization algorithms to find the best complete data pipeline.
	* <ins>*Data analysis and data product backlog*</ins>.
	* <ins>*Priority: 4*</ins>.

2. Model evaluation.

* Inference procedures: bootstrap or sampling techniques to calculate means and standard deviations of performance metrics, so uncertainty can be assessed when comparing different pipelines and models.
	* <ins>*Data analysis and data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.
	* For example: T estimations (full samples with replacement or not) and test data evaluations, then definition of best models according to the average of performance metrics or statistics of test.
* Operational metrics: raise assumptions to define and calculate operational metrics.
* Threshold definition based on operational metrics.
treatment.
	* <ins>*Data analysis backlog*</ins>.
	* <ins>*Priority: 2*</ins>
* Analysis of predictions.
	* Major errors: define further investigations after finding the highest prediction errors.
		* <ins>*Data analysis backlog*</ins>.
		* <ins>*Priority: 3*</ins>.
	* Apply model agnostic explanations for major errors (false positives, false negatives).
		* <ins>*Data analysis backlog*</ins>.
		* <ins>*Priority: 5*</ins>.

3. Final model training.

4. Explainability and interpretability.
* Create a class that produces explanations for predictions.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 5*</ins>.

* Further investigation of which variables could be removed from the set of inputs in order to improve model performance.
* Define a broader range of possible values for main hyper-parameters of the following algorithms: random forest ("min_samples_split") and SVM ("C").
* Define additional hyper-parameters for optimization for each learning algorithm.
	* <ins>*Data analysis backlog*</ins>.
	* <ins>*Priority: 3*</ins>.


### Deployment<a id='deploy'></a>
Using model artifacts, an application and API are created so users can consume the model. Besides, all deployment infrastructure is configured.

1. Application and API development.

* User interface.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 4*</ins>.

* Create an illustration with the application design.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.

2. Deployment infrastructure.

* Create and use a Docker Compose file instead of Dockerfile for deployment.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.

* Model serving.
	* Create a stronger security group for the ML app server.
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 5*</ins>.

	* Deployment with Docker and Elastic Beanstalk.
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 3*</ins>.

	* Deployment with Amazon Lambda.
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 3*</ins>.

	* Implementation of CI/CD tools.
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 5*</ins>.

	* Configure ELB and ASG for distributing traffic and managing a cluster of servers, respectively (server-based architecture with Docker, EC2 and S3).
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 1*</ins>.

	* Create User Data configuration for standardized instance launch (server-based architecture with Docker, EC2 and S3).
		* <ins>*Data product backlog*</ins>.
		* <ins>*Priority: 1*</ins>.

3. Model monitoring.

* Define a roadmap of developments to implement both operational and performance monitoring. This involves listing relevant tasks and building codes for running and orchestrating them.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 3*</ins>.

4. Model re-training.

* Create codes that automatically runs data modeling given evidences of concept drifts or deterioration of performance metrics by the current model.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 2*</ins>.

5. Model governance.

* Discuss and write main aspects about model governance, risk management and auditing.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 2*</ins>.

--------------
### Documentation<a id='doc'></a>

* Register, make reference and discuss theoretical and methodological references.
	* <ins>*Data product backlog*</ins>.
	* <ins>*Priority: 4*</ins>.
