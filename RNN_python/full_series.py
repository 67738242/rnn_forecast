import pandas as pd
import numpy as np

def Insert_row_(row_number, df, row_value):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]
    # Store the result of lower half of the dataframe
    df2 = df[row_number:]
    # Inser the row in the upper half dataframe
    df1.loc[row_number]=row_value
    # Concat the two dataframes
    df_result = pd.concat([df1, df2])
    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]
    # Return the updated dataframe
    return df_result


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
for j in range(3):
    print('2time')
    i = 0
    for i in range(0, len(dataframe)-2):
        time_i = int(str(dataframe.iloc[i]['date-time'])[-8:-6])
        time_i_1 = int(str(dataframe.iloc[i + 1]['date-time'])[-8:-6])
        day_i = str(dataframe.iloc[i]['day'])
        day_i_1 = str(dataframe.iloc[i + 1]['day'])
        day_data_i = dataframe.iloc[i].day_data
        day_data_i_1 = dataframe.iloc[i + 1].day_data

        # miss_val = (int(dataframe.iloc[i-23]['number']) + \
        #     int(dataframe.iloc[i-23-24]['number']) + \
        #     int(dataframe.iloc[i-23-48]['number'])) / 3

        if time_i != 23 and time_i + 1 != time_i_1:
            date_time_i = pd.to_datetime(str(dataframe.iloc[i]['date-time'])[:10] + ' ' + str(time_i + 1)+':00:00')

            dataframe = Insert_row_(i + 1, dataframe, [day_data_i, day_i, date_time_i, pd.np.nan])
            print(time_i, time_i_1)
            print(i)

        elif time_i == 23 and time_i_1 != 0:
            date_time_i_1 = pd.to_datetime(str(dataframe.iloc[i + 1]['date-time'])[:10] + ' 0:00:00')

            dataframe = Insert_row_(i + 1, dataframe, [day_data_i_1, day_i_1, date_time_i_1, pd.np.nan])

            print(time_i, time_i_1)
            print(i)

dataframe = dataframe.set_index(['date-time'], drop=True)
dataframe = dataframe.interpolate('time')
dataframe = dataframe.reset_index()
dataframe.loc[(dataframe['day_data'] == 'holi') | \
    (dataframe['day'] == 'Sun') | \
    (dataframe['day'] == 'Sat'), \
    'binari']=1

dataframe.loc[~((dataframe['day_data'] == 'holi') | \
    (dataframe['day'] == 'Sun') | \
    (dataframe['day'] == 'Sat')), \
    'binari']=0

dataframe = dataframe.set_index(['day_data', 'day', 'date-time'], drop=True)

dataframe.to_excel('/Users/masaki/Dropbox/RNN_python/series_data/dev_num.xlsx')
