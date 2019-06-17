import pandas as pd
import matplotlib.pyplot as plt
import holitest as hlt
import openpyxl
import os
from statistics import mean, median,variance,stdev
import numpy as np

learning_length = 10*24
persentile_num = 95
# error_data_path = '/tmp/RNN_python/input_digits=40output_data_test/seq2seq_error_p_h.xlsx'
mape=[]
mape_path = '/tmp/RNN_python/mape/'
os.makedirs(mape_path, exist_ok=True)

error_data_path = '/tmp/RNN_python/output_data_sarima/s_arima_err_p_h_data1.xlsx'
error_p_h_data = pd.read_excel(
    error_data_path,
    index_col=[0,1,2],
    header=0
)

eval_data_set_kari = hlt.eval_series_data()
eval_data_set = eval_data_set_kari[learning_length:]
real_num = eval_data_set.values

error_p_h = abs(error_p_h_data.values)
length = len(error_p_h_data)

for i in range(len(error_p_h_data)):
    mape.append(error_p_h[i]/real_num[i])

error_p_h = sorted(np.reshape(error_p_h, -1))
# print(error_p_h_data)
persentile_arr = round(length * persentile_num/100)
persentile_val = error_p_h[persentile_arr]
# print(error_p_h_data)
# print(persentile_val)
# print(mape)
mape = np.array(mape)
sort_mape = sorted(mape)
print(mape.mean(), mape.var(), sort_mape[persentile_arr])
mape = np.reshape(mape, -1)
mape_data = pd.DataFrame(mape, index=error_p_h_data.index)
# mape_data.to_excel(mape_path + 'mape.xlsx')
# mape_data.to_excel('mape.xlsx')
