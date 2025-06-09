import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft
from scipy import signal
from numpy import array
# from numpy import fft
from scipy.signal import spectrogram
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve


# 绘制降雨阴影区域
def shaded_area(method_index, threshold, time):  # method_index数据格式应为np.array
    cml_wet = method_index > threshold

    # Get start and end of dry event
    wet_start = np.roll(cml_wet, -1) & ~cml_wet
    wet_end = np.roll(cml_wet, 1) & ~cml_wet

    # Plot shaded area for each wet event
    for wet_start_i, wet_end_i in zip(
            wet_start.nonzero()[0],            # np.array格式才有nonzero函数
            wet_end.nonzero()[0],
    ):
        plt.axvspan(time[wet_start_i], time[wet_end_i], color='b', alpha=0.1)
    return wet_start, wet_end, cml_wet


# 制作混淆矩阵
def cml_confusion_matrix(rainfall_len, cml_wet):
    y_true = []
    y_pred = []
    for i in range(len(rainfall_len)):
        if rainfall_len.values[i] != 0:
            y_true.append(1)
        else:
            y_true.append(0)

        y_pred.append(int(cml_wet[i]))

    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confusion matrix:\n', cm_norm)
    plt.matshow(cm_norm, cmap=plt.cm.Greens)
    # plt.colorbar()
    fmt = '.2%'

    thresh = str(cm_norm.max() / 2)  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(len(cm)):
        for y in range(len(cm)):
            num = format(cm_norm[x, y], fmt)
            plt.annotate(f'{cm[x, y]}\n({num})', xy=(y, x), verticalalignment='center', horizontalalignment='center',
                         fontsize=18, color="white" if cm_norm[x, y] > float(thresh) else "black")
    # plt.figure(figsize=(6, 6))
    # plt.title('RSD', fontsize=18)
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(range(0, 2), labels=['dry', 'wet'], fontsize=18)
    plt.yticks(range(0, 2), labels=['dry', 'wet'], rotation=90, fontsize=18)
    plt.show()

    # 计算一级指标tn, fp, fn, tp；二级指标Accuracy, Precision, Sensitivity,Specificity; 三级指标F1-Score；
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tpr = tp / (tp + fn)
    specificity = tn / (fp + tn)
    fpr = 1 - specificity
    F1 = 2 * precision * sensitivity / (precision + sensitivity)

    print(tn, fp, fn, tp)
    print('accuracy=', accuracy)
    print('precision=', precision)
    print('recall=sensitivity=TPR: ', tp / (tp + fn))
    # print('specificity: ', tn / (tn + fp))
    print('FPR: ', fpr)
    print('F1-Score =', F1)
    return y_true


def pr_curve(method_index, y_true, threshold):
    # 计算y_scores（推荐方法3）
    k = 0.5
    y_scores = 1 / (1 + np.exp(-k * (method_index - threshold)))

    # 计算AUC等指标
    print("ROC AUC:", roc_auc_score(y_true, y_scores))

    # 可视化PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.show()
    return y_scores


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'(AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    print('auc:', roc_auc)
    # print(fpr, tpr, thresholds)
    plt.show()


# 定义计算功率谱密度函数,返回功率谱密度和频率
def power_spectrum_density(data, L, win, fs, nfft):
    pxx1 = []
    for t in range(L, int(len(data))-L):
        stft = fft(win*array(data)[t-L:t+L])    # 计算短时傅里叶变换(变换区间为data[0, N-1])
        p = 2*np.abs(stft)**2/fs/np.linalg.norm(win)**2   # 计算功率谱密度（单边谱要乘以2）
        pxx1.append(p)                                    # 维度：(N-nfft+1, nfft)
    freq = np.array([k*fs/nfft for k in range(nfft//2)])     # 计算单边频率轴
    pxx2 = []
    for j in range(len(pxx1)):
        pxx_k = pxx1[j][:nfft//2]
        pxx2.append(pxx_k)     # 只取前一半的功率谱密度（与频率一一对应）Pxx2维度为(N-nfft+1,128)
    return pxx2, freq


# 计算每个窗口低频和高频信号的平均功率谱密度之差
def psd_sum_diff(Pxx_cml_norm, freq_cml, f_divide):
    Pxx_sum_diff = []
    for Pxx_i in range(len(Pxx_cml_norm)):  # Pxx_i取值范围[0:N-nfft+1]
        Pxx_k = Pxx_cml_norm[Pxx_i]  # 取其中的一行，一行有128个值
        Pxx_sum_low = 0
        Pxx_sum_high = 0
        N_low = 0
        N_high = 0
        for f_i in range(len(freq_cml)):  # f_i取值范围[0:128]
            if freq_cml[f_i] <= f_divide:
                Pxx_sum_low += Pxx_k[f_i]  # Pxx_k中前
                N_low += 1
            else:
                Pxx_sum_high += Pxx_k[f_i]
                N_high += 1
        Pxx_sum_low = Pxx_sum_low / N_low
        Pxx_sum_high = Pxx_sum_high / N_high
        p_diff = Pxx_sum_low - Pxx_sum_high
        Pxx_sum_diff.append(p_diff)  # 维度N-nfft+1*1
    # print(N_low, N_high)
    # print(Pxx_sum_diff[0:3])
    # print(min(Pxx_sum_diff))
    # print(max(Pxx_sum_diff))
    # print('len(Pxx_sum_diff)', len(Pxx_sum_diff))   # N-nfft+1
    # print(Pxx_sum_diff[0:3])                        # 输入前三项的值
    return Pxx_sum_diff

