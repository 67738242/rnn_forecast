import pandas as pd
import numpy as np


df_traffic_data = pd.read_excel(
    # 'C:/Users/kenta/Dropbox/RNN_python/series_data/query_result_evl.xlsx',
    '/Users/masaki/Dropbox/RNN_python/series_data/query_result_evl.xlsx',
    # '/Users/masaki/Dropbox/RNN_python/series_data/query_result_attri.xlsx',
    # path_data.resolve(),
    # '/tmp/RNN_python/series_data/query_result_attri.xlsx',
    # '/tmp/RNN_python/series_data/query_result_evl.xlsx',
    columns = [0,1,2],
    # columns = [0,1],
    header = 0
    )

dataframe = df_traffic_data.reset_index(drop=True)

c = 0
drop_day = []
drop_n = []

for i in range(0, len(dataframe)-1):
    if int(str(dataframe.iloc[i]['date-time'])[-8:-6]) != 23 and \
        int(str(dataframe.iloc[i]['date-time'])[-8:-6]) + 1 != int(str(dataframe.iloc[i + 1]['date-time'])[-8:-6]):
        dataframe.index.insert(i, [, , int(str(dataframe.iloc[i]['date-time'])[-8:-6]) + 1, 7777])
        print(i)

        print(str(dataframe.iloc[i]['date-time'])[-8:-6], str(dataframe.iloc[i+1]['date-time'])[-8:-6])

    elif (dataframe.iloc[i]['day'] != dataframe.iloc[i+1]['day']) and (c != 23):
        drop_day.append(i - c)
        drop_n.append(i)
        c = 0

    elif (dataframe.iloc[i]['day'] != dataframe.iloc[i+1]['day']) and (c == 23):
        c = 0
