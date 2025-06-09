import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 读取数据
cml_rsl = pd.read_excel('../generate_file/cml_rsl_processed.xlsx')
cml_rsl['TIME'] = pd.to_datetime(cml_rsl['TIME'])  # 确保 'TIME' 列是 datetime 类型
cml_rsl.set_index('TIME', inplace=True)  # 将 'TIME' 设为索引

# 按时间范围切片
start_time = datetime.datetime(2024, 7, 10, 0, 0)
end_time = datetime.datetime(2024, 10, 10, 23, 59)
cml_rsl = cml_rsl[start_time:end_time]  # 现在可以正确切片
# print(cml_rsl)   # 133920
cml_rsl = cml_rsl.reset_index()  # 或cml_rsl.reset_index(inplace=True)
# print(cml_rsl)
# RG = pd.read_excel('../experiment_data/RG_01_3month.xlsx')


# Hampel滤波
def hampel_filter(data, window_size=3, threshold=3.0):
    """
    Args:
        data: 输入时序数据（Pandas Series或NumPy数组）
        window_size: 滑动窗口大小（建议奇数，默认3）
        threshold: 异常检测阈值（默认3.0，类似3σ）
    Returns:
        cleaned_data: 过滤后的数据
        anomalies: 异常值的索引和原始值
    """
    # 参数校验
    assert window_size % 2 == 1, "window_size应为奇数"
    assert window_size > 0, "window_size需为正整数"

    k = 1.4826  # 正态分布下的MAD缩放系数
    cleaned_data = data.copy()
    anomalies = []
    half_window = window_size // 2

    for i in range(len(data)):
        # 对称窗口切片
        window = data[max(0, i - half_window): min(len(data), i + half_window + 1)]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))

        if np.abs(data[i] - median) > threshold * mad:
            anomalies.append((i, data[i]))
            cleaned_data[i] = median  # 用中位数替换异常值

    return cleaned_data, anomalies


cleaned_data, anomalies = hampel_filter(cml_rsl['RSL'])
# print(cleaned_data)
# print(type(cleaned_data))
# print("异常点位置和值:", anomalies)
cml_rsl_cleaned = pd.DataFrame({'TIME': cml_rsl['TIME'], 'RSL': cleaned_data})
# print(cml_rsl_cleaned)   # 133920
cml_rsl_cleaned.to_excel('../processed_data/cml_rsl_cleaned_3mon.xlsx', index=False)


