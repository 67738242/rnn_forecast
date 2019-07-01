import pandas as pd
import holitest as hlt
import matplotlib.pyplot as plt
from scipy.stats import norm
import openpyxl
import os
from statistics import mean, median,variance,stdev
import numpy as np

error_p_h_data = pd.read_excel(
    path_error,
    header=0,
    index_col=[0,1,2]
)

error_p_h = error_p_h_data.values.reshape(-1)

lkhd = []

for i in range(len(error_p_h)):
    lkhd.append(norm.pdf(error_p_h[i], loc = 0, scale = 30))

print(lkhd)

lkhd_data= pd.DataFrame(lkhd)
lkhd_data.to_excel('/tmp/RNN_python/evaluate_data1/lkhd_p_h.xlsx')
