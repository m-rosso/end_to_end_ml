## End-to-end ML

This repository has codes, data and artifacts developed in a project whose goal is to build a **complete data product** based on a machine learning model for predicting whether a mobile app is a malware or not. This application framework follows from this [dataset](https://www.kaggle.com/datasets/saurabhshahane/android-permission-dataset) containing 184 columns that are attributes for 27310 rows, each consisting of a single mobile app.

Constructing a prototype of a complete data product demands engaging in all activities that go from business understanding (discussion) to model development, converging to a deployment stage. This goal reflects the following **objectives**:
* Integrate different study fields.
* Apply several distinct technologies.
* Improve project management skills.
* Implement a deployment solution as realistic as possible.

Consequently, these were the main **technologies** used in this project, which are reflected in codes of *notebooks* and *src* folders and that are mostly present in *requirements.txt* file.

|       **Task**      |                    **Technology**                    |
|:-------------------:|:----------------------------------------------------:|
| Data management     | [pandas](https://pandas.pydata.org/docs/)<br> [numpy](https://numpy.org/doc/)                                     |
| Machine learning    | [scikit-learn](https://scikit-learn.org/)<br> [LightGBM](https://lightgbm.readthedocs.io/)<br> [XGBoost](https://xgboost.readthedocs.io/)<br> [scipy](https://docs.scipy.org/doc/scipy/)      |
| Deployment          | [FastAPI](https://fastapi.tiangolo.com/)<br> [Docker](https://www.docker.com/)<br> [DockerHub](https://hub.docker.com/)<br> [Postman](https://www.postman.com/)<br> [Amazon EC2](https://aws.amazon.com/pt/ec2/)<br> [Amazon S3](https://aws.amazon.com/pt/s3/) |
| Project development | [VS Code](https://code.visualstudio.com/)<br> [Google Colab](https://colab.research.google.com/)                             |

Throughout stages of traditional [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology for data science projects, the following **artifacts** were built:

|     **Stage**    |                                                                                                               **Artifacts**                                                                                                               |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Discussion       | Assumptions, definitions and objectives.                                                                                                                                                                                                  |
| Data engineering | Training and test datasets.                                                                                                                                                                                                               |
| Data preparation | Python modules and codes for using their functions and classes to prepare data during experimentation in data modeling stage.                                                                                                             |
| Data modeling    | Pickle file with a complete data pipeline that transforms a raw input data point and returns a predicted probability.                                                                                                                       |
| Deployment       | Python script for running the application + API that allow a user to consume predictions from the model.<br> Dockerfile for creating a Docker image and, consequently, containers so the application + API can start running. |
| Documentation    | Markdown files describing all activities implemented in each of the above stages of CRISP-DM methodology.                                                                                                                                 |

More details regarding development can be found in the *docs* folder and in the [Github page](https://github.com/m-rosso/end_to_end_ml/wiki) of this repository, which consist of the **project documentation**.
