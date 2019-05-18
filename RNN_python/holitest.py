import pandas as pd
import numpy as np

def eval_series_data():
    df_traffic_data = pd.read_excel(
        # 'C:/Users/kenta/Dropbox/RNN_python/series_data/query_result_evl.xlsx',
        # '/Users/masaki/Dropbox/RNN_python/series_data/query_result_evl.xlsx',
        # '/Users/masaki/Dropbox/RNN_python/series_data/query_result_attri.xlsx',
        # path_data.resolve(),
        # '/tmp/RNN_python/series_data/query_result_attri.xlsx',
        '/tmp/RNN_python/series_data/query_result_evl.xlsx',
        columns = [0,1,2],
        # columns = [0,1],
        header = 0
        )

    data = df_traffic_data[((df_traffic_data['day'] != 'Sun') & \
        (df_traffic_data['day'] != 'Sat'))& (df_traffic_data['day_data'] != 'holi')]

    dataframe = data.reset_index(drop=True)

    # print(len(dataframe))

    c = 0
    drop_day = []
    drop_n = []

    for i in range(0, len(dataframe)-1):
        if (dataframe.iloc[i]['day'] == dataframe.iloc[i+1]['day']):
            c += 1

        elif (dataframe.iloc[i]['day'] != dataframe.iloc[i+1]['day']) and (c != 23):
            drop_day.append(i - c)
            drop_n.append(i)
            c = 0

        elif (dataframe.iloc[i]['day'] != dataframe.iloc[i+1]['day']) and (c == 23):
            c = 0

    drop_day_ = np.array(drop_day)
    drop_n_ = np.array(drop_n)

    for h in range (len(drop_day)):
        for k in range(drop_day_[h], drop_n_[h] + 1):
            dataframe = dataframe.drop(k)

    dataframe = dataframe.reset_index(drop=True)

    for i in range(len(dataframe)-1):
        if dataframe.iloc[i]['day_data'] == 'TRUE':
            dataframe = dataframe.drop(j for j in range(i, 24 + i))

    # print(dataframe[24 * 5:])
    # print(len(dataframe))

    df2 = dataframe.set_index(['day_data', 'day','date-time'], drop=True)

    # print(df2)
    return df2
