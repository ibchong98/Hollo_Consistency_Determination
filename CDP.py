import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmdoels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#transfer data
import train_model.py #Import the train model 

#Data which will be tested

input_data = pd.read_csv('input_file_here') #test data

tested = ARIMA(input_data, order=(0,1,1)) #parameters should be changed according to the preprocessing stage

tested = tested.fit(trend='c',full_output=True, disp=1) #parameter should be changed samely

def consistency_determ(train_model, tested):
    
