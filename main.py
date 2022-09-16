##############################################################################
##############################################################################
###################################LIBRARIES##################################

__author__ = 'm-rosso'

import pandas as pd
import json
import pickle
from copy import deepcopy
import uvicorn
from fastapi import FastAPI
import boto3
from dotenv import load_dotenv

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), 'src'
        )
    )
)

##############################################################################
##############################################################################
###########################FUNCTIONS AND CLASSES##############################

from utils import predict_label
from production import UserRequestIn

##############################################################################
##############################################################################
###########################DATA AND CONFIGS###################################

# App configuration:
with open('config/app_configs.json', 'r') as json_file:
    CONFIGS = json.load(json_file)

# Training data:
df_train = pd.read_csv('artifacts/df_train.csv', dtype={'app_id': int})

# Object of fitted pipeline:
model = pickle.load(open('artifacts/model.pickle', 'rb'))

# AWS credentials:
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Connection with AWS API:
client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

##############################################################################
##############################################################################
##################################APP#########################################

# Creating the API object:
app = FastAPI()


@app.post('/predict')
def predict(user_request: UserRequestIn):
    # Extracting API data:
    request_id = user_request.request_id
    input_data = user_request.input_data

    try:
        # Predicted probability of malware:
        prediction = model.predict(
            input_data=input_data,
            training_data=df_train
        )
        score_pred = prediction[0]

        # Predicted class:
        label_pred = predict_label(
            score=score_pred, threshold=CONFIGS['threshold'],
            labels=['safe', 'malware']
        )

        # API response message:
        resp_msg = predict_label(
            score=score_pred, threshold=CONFIGS['threshold'],
            labels=[
                CONFIGS['responses']['safe'],
                CONFIGS['responses']['malware']
            ]
        )

    except Exception as error:
        score_pred = None
        label_pred = None
        resp_msg = f'The application returned the following exception: {str(error)}'

    # API response:
    response = {
        'request_id': request_id,
        'score_pred': score_pred,
        'label_pred': label_pred,
        'response_message': resp_msg,
        'comment': 'Deployed using FastAPI and Docker.'
    }

    # Storing request and response data:
    req_resp_data = deepcopy(response)
    req_resp_data['input_data'] = input_data
    req_resp_data = json.dumps(req_resp_data, ensure_ascii=False)
    client.put_object(
        Bucket='ml-app-inputs-preds',
        Key=request_id+'.json',
        Body=req_resp_data
    )

    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
