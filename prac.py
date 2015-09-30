import pandas as pd
import numpy as np
import csv as csv
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('train.csv',header=0)

data['Gender'] = data['sex'].map({'female:'0,'male':1}).astype(int)
