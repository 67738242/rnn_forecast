import holitest as hlt
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import os
from statistics import mean, median,variance,stdev
import numpy as np

eval_data_set_kari = hlt.eval_series_data()

error_ph_data = pd.read_excel(
    '/tmp/RNN_python/evaluate_data1/lkhd_p_h.xlsx',
    header=0,
    index_col=0
    )

error_ph = error_ph_data.values

error_ph_log = np.log10(error_ph)

error_index = np.where(error_ph_log < -5)
print(error_index)
