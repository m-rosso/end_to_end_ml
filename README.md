## End-to-end ML

This repository has codes, data and artifacts developed in a project whose objective is to build a *complete data product* based on a machine learning model for predicting whether a mobile app is a malware or not. This application framework follows from this [dataset](https://www.kaggle.com/datasets/saurabhshahane/android-permission-dataset) containing 184 columns that are attributes for 27310 rows, each consisting of a single mobile app.

Throughout stages of traditional [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology for data science projects, the following artifacts were built:

|     **Stage**    |                                                                                                               **Artifacts**                                                                                                               |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Discussion       | Assumptions, definitions and objectives.                                                                                                                                                                                                  |
| Data engineering | Training and test datasets.                                                                                                                                                                                                               |
| Data preparation | Python modules and codes for using their functions and classes to prepare data during experimentation in data modeling stage.                                                                                                             |
| Data modeling    | Pickle file with a complete data pipeline that transforms a raw input data point and returns a predicted probability.                                                                                                                       |
| Deployment       | Python script for running the application + API that allow a user to consume predictions from the model.<br> Dockerfile for creating a Docker image and, consequently, containers so the application + API can start running. |
| Documentation    | Markdown files describing all activities implemented in each of the above stages of CRISP-DM methodology.                                                                                                                                 |

More details regarding the development can be found in the *docs* folder and in the [Github page](https://github.com/m-rosso/end_to_end_ml/wiki) of this repository, which consist of the project documentation.
