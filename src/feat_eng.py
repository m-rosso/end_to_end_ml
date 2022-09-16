####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function for calculating the number of related apps that can be found in historical data:

def known_related_apps(data: pd.DataFrame, related_apps: str):
  """
  Function for calculating the number of related apps that can be found in historical data.

  :param data: historical data of reference.
  :type data: dataframe.
  :param related_apps: text containing the collection of related apps for a given app.
  :type related_apps: string.

  :return: number of known related apps.
  :rtype: integer.
  """
  if pd.isna(related_apps):
      return np.NaN
  else:
      return sum([1 if a in list(data['package']) else 0 for a in related_apps.split(', ')])

####################################################################################################################################
# Function for calculating the number of related apps that are malwares:

def related_malwares(data: pd.DataFrame, related_apps: str):
  """
  Function for calculating the number of related apps that are malwares.

  :param data: historical data of reference.
  :type data: dataframe.
  :param related_apps: text containing the collection of related apps for a given app.
  :type related_apps: string.

  :return: number of known malware related apps.
  :rtype: integer.
  """
  if pd.isna(related_apps):
      return np.NaN
  else:
      return sum([1 if np.nanmean(data[data['package']==a]['class']) > 0 else 0 for a in related_apps.split(', ')])
