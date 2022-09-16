## Deployment
This stage of a machine learning project takes artifacts built during previous stages and integrates them into an infrastructure so the analytical solution can be consumed in a productive context. One standard deployment strategy embeds the solution (e.g., a model or dashboard) into an application and creates an API through which the user can join this application to its process or system. Besides, if necessary a user interface can be developed so she/he can directly interact with the application.

In addition to this **application and API development**, the **infrastructure** is another crucial aspect of any deployment architecture. How codes and data should be packaged and provided? How the application is going to be hosted so the user can access the analytical solution whenever needed? These are some main questions that should be answered when deploying an application.

Once the analytical solution has been deployed, its functioning needs to be under constant **monitoring**, both in terms of operation and performance. Eventually, the solution should be calibrated to reflect updated circumstances. For instance, given a predictive or clustering task, **model re-training** may be implemented periodically. Finally, **model governance** should be on in order to assess and manage risks that can be raised by the consumption of the analytical solution.

--------------
### Application and API development
Irrespective of the complexity of the productive context where a constructed analytical solution is about to work, it should be wrapped in an **application** that will make possible to users to make use of it, or even as a building block of a broader application that already can run without it, but that will greatly benefit from its integrated operation. Besides, an **API** is most likely the best way to connect users to the application from where a model is running.

#### Application
The application is built from codes that *combine a fitted model with an engine that takes a raw input data point and returns a prediction or an exception*. The Python library used to develop this application is [FastAPI](https://fastapi.tiangolo.com/), which also contributes to make an API available for consuming the app (see the next subsection for more on this). In order to provide a front-end to the application, additional codes could be developed and integrated (see the final subsection about user interface).

The application development is centered in a **Python script**, named here as *main.py*. This file imports all necessary libraries, functions and classes. Next, it also reads the data needed to apply the data preparation pipeline, the model object and all configurations so the app can run as expected.

The **core of this script** is the application itself, which is created by instantiating an object of the *FastAPI* class. This object is used as a *decorator* to a function that holds the inner functioning of the app, which is described by the following steps:
* The application receives a user request, extracts the input data from it and applies the model object to create a **prediction** for the probability of the event where the associated app is a malware.
* Next, given this predicted probability and a previously defined threshold, a **predicted label** (malware or safe app) is assigned to the user app.
* Then, it comes the definition of the appropriate **message** that should be sent to the user as a function of the predicted label.
* Finally, the app stores both the received input data and the response sent to the user.

In order to develop the application that would wrap the predictive model constructed during *Data Modeling* stage, a notebook named "Deployment - Production Model" was created. In this notebook, that can be found in the *notebooks* folder of the project, all artifacts generated in the modeling stage were imported and used to declare an **instance of *Model* class** (check *production.py* module in folder *src* for its documentation).

The *Model* class makes use of a data preparation pipeline, an ensemble of fitted models, and some other objects that guide the production of a prediction for a given raw input data point. Then, the created object of *Model* class was saved as a [*pickle*](https://docs.python.org/3/library/pickle.html) file that can be used inside the application and through the API that were developed using FastAPI. Finally, the notebook also implements tests to estimate the latency time for producing predictions and to develop errors handling.

#### API
The API (Application Programming Interface) provides **user access to an endpoint for consuming the application**. [FastAPI](https://fastapi.tiangolo.com/) allows the combined creation of an application with its API configuration when [uvicorn](https://www.uvicorn.org/) library is used along. This library connects the developed application with the [host address](https://www.computernetworkingnotes.com/networking-tutorials/ip-address-network-address-and-host-address-explained.html) and the [port](https://en.wikipedia.org/wiki/Port_(computer_networking)) where it should be run.

The API created makes use of a FastAPI object as a decorator to the function that holds the application inner functioning. More precisely, a method of that object is called, which consists of any main [REST API method](https://restfulapi.net/http-methods/) (e.g., get, post or put). In this project, the **post method** was used, so whenever a user appropriately interacts with the API a post method is being executed.

After a request has been sent to the application using the API, it works as described in the subsection above and returns predictions for the app associated with the user request. Then, the **API response** that is sent to the user contains: *request id*, *predicted probability*, *predicted label*, and *response message*. Note that in the case of any error, the exception is returned together with assigned values for the predictions of score and label.

*For more on the API developed and how to consume it, check the the Python script named api_tests in the tests folder.*

#### User interface
How the user is going to interact with the application consists of the **front-end** of a product. *Even though it is highly relevant, this task is not covered by this project and can be found in the backlog for future developments.*

--------------
### Deployment infrastructure
The application and API that constitute a model deployment relies on an infrastructure in order to them be effectively consumed by users. This infrastructure should be managed at least in a high-level by the project development team. How the code and data will be packaged and delivered for deployment and how the application and API will be hosted are two examples of fundamental infrastructure issues to be discussed.

#### Code packaging
Once data modeling has been implemented, with all rounds of experimentation and the final model training, and once the application and API has also been constructed, it is time to move **from development stage to production stage**, which requires collecting all codes and data needed for deployment.

Artifacts should move from a development environment, such as a local/remote machine connected or not to a Github account and maintained using VS Code, for instance, to a production environment, such as a cloud server. Consequently, it is important **to package codes and data** so they can migrate more easily (ideally, in an automatic way - see the topic on CI/CD below).

The standard method of application packaging involves [**Docker**](https://medium.com/geekculture/introduction-to-docker-and-container-based-development-microservices-part-1-f41522e91d4a), a technology that allows code to run in any machine irrespective of its operational system and dependencies. Deployment with Docker involves applying the following steps:
* Creation of a **Dockerfile**, which is a recipe indicating where (i.e., the [image](https://en.wikipedia.org/wiki/System_image) and its configuration) and how an application should be run (all steps up to executing the code).
* Building a **Docker image** from the Dockerfile, so the application environment can be set up.
* Registering the Docker image using **DockerHub**.
* Creating and running a **Docker container** upon the Docker image so the application starts working.

Ideally, any modification in codes should immediately affect the application deployed, the so-called principle of [continuous integration, continuous delivery (CI/CD)](https://en.wikipedia.org/wiki/CI/CD). *Although this is a highly important topic for an end-to-end ML project, the implementation of CI/CD tools was inserted in the backlog with top-priority.*

#### Model serving
The deployment strategy that uses Docker to package and run codes that wrap a constructed machine learning model can make use either of **local or remote servers**. In this project, the deployment was implemented in both of these alternatives. Even so, deployment in the cloud is more similar to what is expected to be found in real-world scenarios.

After launching an **EC2 instance** in the AWS management console and packaging code (either by creating a Docker image or using protocols such SSH or PSCP to transfer the project), some settings must be done, such as installing Docker and Nginx (for IP routing, so API calls can be sent to the EC2 instance IP). Next, a Docker container is created and run, so the application starts working in the cloud. Note that all this operations cloud be inserted into the *EC2 user data* so they are implemented right after the instance is launched. However, the instance setup and application deployment were made by *connecting to the remote server using SSH/Putty*.

Another AWS service used was **Amazon S3**, since the created application has a block of code that integrates with an AWS account and makes API calls to put objects (JSON file with input data point and prediction regarding malware status) using the Python SDK (*boto3*), which takes place right before the API returns the response to the user that has sent the input data point.

Therefore, this deployment strategy, that combines FastAPI, Docker, EC2 and S3, is a non-serverless architecture. *An improvement of this strategy that is present in the backlog involves using other AWS services, ELB and ASG, for distributing traffic and managing a cluster of servers, respectively.* Additionally, *some additional items in the backlog consist of serverless architectures, such as deployment with Elastic Beanstalk, and deployment with Amazon Lambda.*

--------------
### Model monitoring
After an analytical solution, such as model, has been deployed, it is necessary to keep track of its functioning. An **operational monitoring** is crucial to evaluate the API responses to the requests sent to the API that holds the application. Some relevant questions to be aware of are the following:
* Do the predictions respect the target variable domain?
* How many errors is the API returning, and which are their causes?
* Is the inner functioning of the application as expected? Is it appropriately saving input data points and the respective predictions?
* Is the latency of responses evolving accordingly?

Also a **performance monitoring** should be constructed to identify changes in *probability distributions (concept drifts)* of inputs ($P(X)$) and the target variable ($P(Y|X)$). Some possible monitoring tasks are listed below:
* Data type of input variables.
* Share of missing values by input variable.
* Descriptive statistics and distribution comparison for input variables.
* Collection of true target values for data points that were subject to model scoring.
* Calculation of performance metrics (specially model accuracy) by opposing true target values against predictions.

*Activities regarding model monitoring were not implemented in this project, but are present in the backlog for future developments.*

--------------
### Re-training
Once a business/operational demand for model training has been declared, or given evidences of concept drifts or deterioration of performance metrics by the current model, it should be replaced by a new model constructed through new rounds of experimentations or just a re-fitting of the model (which may or not include the definition of new values for hyper-parameters).

*Activities regarding model re-training were not implemented in this project, but are present in the backlog for future developments.*

--------------
### Model storage, management, and governance
In real-world applications of machine learning models, third-party teams are assigned to audit the construction and functioning of any analytical solution. Consequently, **model governance** is fundamental to raise confidence on the constructed model. Documentation showing that best practices of data modeling were respected and a good project structure and management will help testifying that the model in production is such that its underlying risks are under control.

*Activities regarding model governance were not implemented in this project, but are present in the backlog for future developments.*
