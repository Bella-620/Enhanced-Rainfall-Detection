import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import datetime


RG = pd.read_excel('../processed_data/RG_01_07_11.xlsx')
# print(len(RG_01))  # 207360

RG.RAIN = pd.Series([0 if abs(x) > 50 else x for x in RG.RAIN])
RG.RAIN = RG.RAIN.rolling(window=10, center=True).mean()
RG.RAIN = RG.RAIN.bfill()
RG.RAIN = RG.RAIN * 60
RG.set_index('TIME', inplace=True)  # 将 'TIME' 设为索引

# 按时间范围切出3个月的数据
st = datetime.datetime(2024, 7, 10, 0, 0)
et = datetime.datetime(2024, 10, 10, 23, 59)
RG_3mon = RG[st:et]
RG_3mon = RG_3mon.reset_index()
RG_3mon.to_excel('../processed_data/RG_3mon.xlsx', index=False)

