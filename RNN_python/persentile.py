import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import statsmodels.api as sm
import os
import scipy.signal as sig
import scipy.io.wavfile as scw
from statistics import mean, median,variance,stdev
import numpy as np

persentile_num = 95
# data_path = '/tmp/RNN_python/input_digits=40output_data_test/seq2seq_error_p_h.xlsx'

data_path = '/tmp/RNN_python/output_data_sarima/s_arima_err_p_h_data1.xlsx'
error_p_h_data = pd.read_excel(
    data_path,
    index_col=[0,1,2],
    header=0
)

length = len(error_p_h_data)

error_p_h_data = sorted(error_p_h_data.abs_num.values)
print(error_p_h_data)
persentile_arr = round(length * persentile_num/100)

persentile_val = error_p_h_data[persentile_arr]
print(persentile_val)
