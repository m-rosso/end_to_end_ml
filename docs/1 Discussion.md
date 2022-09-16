## Discussion
This stage of a machine learning project development, which is the crucial one, was not fully undertaken here given that the **main objectives of this project** are methodological and technological, being defined as these:
* Integrate different study fields.
* Apply several distinct technologies.
* Exercise project management skills.
* Construct a full machine learning product, i.e., an **end-to-end machine learning project**.

Consequently, the following **assumptions** are raised:
* Does not optimize model performance as it would in a business scenario.
* Does not take into account operational metrics.
    * *Business discussion is not part of the project, thus putting the focus on methodological and technological issues*.
* Time information on the apps: in the dataset, there is an absence of creation/registry date for each app.
	* Consequently, it is assumed that the apps data is independent over time.

In short, the goal that this project aims to reach can be summarized as this: **to integrate different aspects of a machine learning system, developing an end-to-end project**.

--------------
### Problem understanding

#### Context
We can imagine the following situation from the [dataset](https://www.kaggle.com/datasets/saurabhshahane/android-permission-dataset) used in this project: once a mobile application is installed on a device, lets call it **AppSafe**, it should provide information regarding the safety of whatever mobile app that is about to be installed thereafter.

Consequently, the *operation under intervention* is mobile apps and the assessment of their safety. Besides, the *situation that final users would be in* can be summarized as follows: *when I download an app, I want to know whether it can cause damages to my device or whether personal data can be exposed, so I can avoid completing installation before cheking the responsibles for the app*.

#### User needs and goals
Hypothetically, the final user is supposed to want to know when an app he/she is about to install is potentially harmful.

#### Data research
We need not to implement this stage as the data was found in Kaggle and can be downloaded [here](https://www.kaggle.com/datasets/saurabhshahane/android-permission-dataset).

#### Objectives
* Operational objectives: the *expected solution* is an app (hypothetically called AppSafe) composed of a model that calculates the risk of a mobile app being a malware and an API that integrates with an app store and with the user by sending him/her a warning message when the mobile app that is about to be downloaded is too risky.
* Data science problem: supervised learning task of a binary classification problem.
* Project objectives: a complete data product (model + API) that would compose a mobile app (AppSafe).

#### Assumptions
Defined by the beggining of this document.

#### Economic issues (risks, costs and benefits)
Do not apply here.

### Domain knowledge research
* *Operational understanding* and *literature review* was not demanded here.
* The *basic entity* of data is given by a mobile app.
* The *target variable* is a flag indicating whether an app is a malware.
* *Attributes* of entities that can help predict the target follow immediately from the original dataset together with additional feature engineering.

### Product/service specification
This **hypothetical AppSafe mobile app** would integrate with app stores, such as [AppStore](https://apple.com/app-store/) or [PlayStore](https://play.google.com/store/), and would collect information on any mobile app available there. Then, by feeding a *machine learning model* with information on an incoming application being downloaded by the user, a probability that this app is a malware would be calculated, and AppSafe could return a warning message to the user if the app has a high risk of causing damages to the device or exposing user sensible data. In short, *the product is an app consisting of a ML model plus an API*.

### Stages and tasks
The project follows the traditional [CRISP-DM](https://pt.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining) methodology, so these are the main stages with their tasks that make the structure of the project and that were implemented throughout the months:
* **Data engineering:** data imports, data understanding and cleaning, exploratory data analysis, feature engineering.
    * *Artifacts:* Jupyter notebook, treated datasets for training and test.
    * *Technologies:* data management libraries: pandas, numpy.
* **Data preparation:** features classification and early selection, data pre-processing (transformations, dataset composition), features selection.
    * *Artifacts:* Jupyter notebook, Python modules with self-made APIs for data preparation.
    * *Technologies:* data management libraries (pandas, numpy), self-made codes for data preparation.
* **Data modeling:** hyper-parameters tuning and model training, model evaluation, model selection and definition of ensemble of models, complete pipeline selection, final model training (fine tuning).
    * *Artifacts:* Jupyter notebooks, Python modules with classes for constructing model ensemble, for producing a prediction from raw instances, and Pickle file of final trained model (object from "Model" class).
    * *Technologies:* different ML libraries (scikit-learn, LightGBM, XGBoost, scipy), self-made experimentation framework.
* **Deployment:** application and API development, development of Dockerfile for building an image with ML application, Docker image registry, model serving simulation (API tests).
    * *Artifacts:* Python module with ML application, Dockerfile, Docker image and container.
    * *Technologies:* FastAPI, Docker and DockerHub, Postman.
* **Documentation, backlog and conclusions**.

### Terms and definitions
A glossary of terms can be developed in the future (this topic is already in the backlog).
