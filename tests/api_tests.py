##############################################################################
##############################################################################
###############################LIBRARIES######################################

import numpy as np
import json
import requests
from time import sleep, time
from copy import deepcopy

##############################################################################
##############################################################################
###########################DATA AND CONFIGS###################################

# Input data:
with open('../artifacts/sample_inputs.json', 'r') as json_file:
    inputs = json.load(json_file)

# API endpoint:
URL = "http://18.231.171.70/predict"

##############################################################################
##############################################################################
#############################API REQUESTS#####################################

for i in range(10):
    # Formatting the required data to be sent to the API:
    sample = np.random.choice(range(len(inputs)), size=1)[0]
    req_id = f'test_{str(int(time()))}'
    INPUT = {'request_id': req_id, 'input_data': inputs[sample]}

    # Sending API request and saving the response:
    req = requests.post(url=URL, json=INPUT)

    # Converting response data into a dictionary:
    response = req.json()
    print(response)
    sleep(3)

##############################################################################
##############################################################################
#########################HANDLING WITH ERRORS#################################

# Absence of attributes:
sample = np.random.choice(range(len(inputs)), size=1)[0]
missing_attr = deepcopy(inputs[sample])
missing_attr.pop('price')

# Irrelevant attributes:
sample = np.random.choice(range(len(inputs)), size=1)[0]
irrel_attr = deepcopy(inputs[sample])
irrel_attr['price2'] = irrel_attr['price']

# Wrong data type:
sample = np.random.choice(range(len(inputs)), size=1)[0]
wrong_type1 = deepcopy(inputs[sample])
wrong_type1['rating'] = str(wrong_type1['rating'])

sample = np.random.choice(range(len(inputs)), size=1)[0]
wrong_type2 = deepcopy(inputs[sample])
wrong_type2['category'] = 10

# Loop over samples of inputs:
for input in [missing_attr, irrel_attr, wrong_type1, wrong_type2]:
    req_id = f'test_{str(int(time()))}'
    INPUT = {'request_id': req_id, 'input_data': input}

    # Sending API requests and saving responses:
    req = requests.post(url=URL, json=INPUT)

    # Converting responses data into dictionaries:
    response = req.json()
    print(response)
    sleep(3)
