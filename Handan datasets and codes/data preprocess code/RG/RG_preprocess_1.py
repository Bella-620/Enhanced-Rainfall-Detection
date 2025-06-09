import pandas as pd
import matplotlib.pyplot as plt


RG_07_09 = pd.read_excel('../original_data/RG_7.10-9.10(30s).xlsx')
RG_07_09['TIME'] = pd.to_datetime(RG_07_09['TIME'])
print(len(RG_07_09))  # 174799
print(type(RG_07_09.TIME[0]))
# RG_.RAIN = [0 if abs(x) > 50 else x for x in RG_02.RAIN]


# 将时间设置为index，按照均匀时间间隔重新索引
def reindex(data, freq):
    date_range = pd.date_range(start=data['TIME'].min(), end=data['TIME'].max(), freq=freq)
    data.set_index('TIME', inplace=True)
    return data.reindex(date_range, method='bfill')


RG_07_09_reindexed = reindex(RG_07_09, '30s')
print(len(RG_07_09_reindexed))  # 180422


# 将30s的数据临近两个求和获得1min的数据
def pair_sum(time, numbers):
    return time[::2], [sum(pair) for pair in zip(numbers[::2], numbers[1::2])]


RG_time, RG_rain = pair_sum(RG_07_09_reindexed.index, RG_07_09_reindexed.RAIN)
RG_07_09_1min = pd.DataFrame({'TIME': RG_time, 'RAIN': RG_rain})
print(len(RG_07_09_1min))  # 90211
RG_07_09_1min.to_excel('../generate_file/RG_01_07_09.xlsx')


RG_09_11 = pd.read_excel('../original_data/RG_9.10-11.30(1min).xlsx')
RG_09_11['TIME'] = pd.to_datetime(RG_09_11['TIME'])
print(len(RG_09_11))  # 94570
RG_09_11_reindexed = reindex(RG_09_11, '1min')
print(len(RG_09_11_reindexed))  # 117150
RG_09_11_reindexed.index.name = 'TIME'
RG_09_11_reindexed = RG_09_11_reindexed.reset_index()  # 将index列变为普通行
print(RG_09_11_reindexed)
RG_09_11_reindexed.to_excel('../generate_file/RG_01_09_11.xlsx')
