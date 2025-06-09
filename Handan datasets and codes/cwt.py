import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy import array

cml = pd.read_excel('../processed_data/cml_rsl_cleaned_3mon.xlsx')
cml['attenuation'] = 27 - cml.RSL
rain = pd.read_excel('../processed_data/RG_3mon.xlsx')
cml['wet'] = rain.RAIN != 0

N = 60  # 分段长度，采样时间间隔1min，60代表60min，即1个小时
L = int(N/2)
total_scale = 20
fs = 1/60


# 定义连续小波变换cwt函数
def my_cwt(data):
    wavelet = pywt.ContinuousWavelet('gaus8')
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * total_scale
    scales = cparam / np.arange(total_scale, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(np.array(data), scales, wavelet, 60)
    return cwtmatr, frequencies


L_sep = int(len(cml)/2)
train_data = cml[:L_sep]   # 需手动
test_data = cml[L_sep:]    # 需手动
# print(train_data)  # 2024-07-10 00:00:00 —— 2024-08-25 11:59:00   length=66960
# print(test_data)   # 2024-08-25 12:00:00 —— 2024-10-10 23:59:00   length=66960


wet_train, dry_train = 0, 0
for i in range(L, len(train_data) - L):
    [cwtmatr_train, frequencies_train] = my_cwt(cml.attenuation[i - L:i + L])
    t_train = np.linspace(i - L, i + L, N)
    plt.contourf(t_train[20:60-20], frequencies_train, abs(cwtmatr_train[:, 20:60-20]))
    plt.axis('off')
    plt.gcf().set_size_inches(256 / 100, 256 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    x_wet_train = '../cwt_pictures/train/wet/''wet-' + str(i) + '.jpg'       # 需手动
    x_dry_train = '../cwt_pictures/train/dry/''dry-' + str(i) + '.jpg'       # 需手动
    if rain.RAIN[i] != 0.0:
        plt.savefig(x_wet_train)
        wet_train += 1
    else:
        plt.savefig(x_dry_train)
        dry_train += 1
    plt.close()
print("wet_train", wet_train)  # 1126
print("dry_train", dry_train)  # 65774


wet_test, dry_test = 0, 0
for i in range(L+L_sep, len(cml) - L):        # 需手动
    [cwtmatr_test, frequencies_test] = my_cwt(cml.attenuation[i - L:i + L])
    t = np.linspace(i - L, i + L, N)
    plt.contourf(t[20:60-20], frequencies_test, abs(cwtmatr_test[:, 20:60-20]))
    plt.axis('off')
    plt.gcf().set_size_inches(256 / 100, 256 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    x_wet_test = '../cwt_pictures/test/wet/''wet-' + str(i) + '.jpg'        # 需手动
    x_dry_test = '../cwt_pictures/test/dry/''dry-' + str(i) + '.jpg'        # 需手动
    if rain.RAIN[i] != 0.0:
        plt.savefig(x_wet_test)
        wet_test += 1
    else:
        plt.savefig(x_dry_test)
        dry_test += 1
    plt.close()
print("wet_test", wet_test)  # 1201
print("dry_test", dry_test)  # 65699

