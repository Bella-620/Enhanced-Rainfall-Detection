import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cml_rsl = pd.read_excel('../original_data/cml-rsl-07-11.xlsx')
print(len(cml_rsl))  # 187733
cml_rsl = cml_rsl.drop_duplicates().reset_index(drop=True)  # 删除所有的重复行，仅保留其中一行
print(len(cml_rsl))  # 187291
print(cml_rsl.TIME.is_monotonic_increasing)   # True
#
#
date_range = pd.date_range(start=cml_rsl['TIME'].min(), end=cml_rsl['TIME'].max(), freq='min')
cml_rsl = cml_rsl.set_index('TIME')    # 或cml_rsl.set_index("TIME", inplace=True)
cml_rsl = cml_rsl[~cml_rsl.index.duplicated(keep='first')]  # 删除重复的索引行（除第一次出现）
# print(cml_rsl)  #  这时index上还有TIME
print(len(cml_rsl))  # 186950


# reindex() 不会自动保留原索引名称，尤其是当新索引（如 date_range）没有名称时
# 解决方案：在创建 date_range 时通过 name="TIME" 指定名称，或
# 在 reindex() 后手动设置 cml_rsl_reindexed.index.name = "TIME"。
cml_rsl_reindexed = cml_rsl.reindex(date_range)  # 使用reindex来匹配新的日期范围
# print(cml_rsl_reindexed)  # 这时index上没有TIME
cml_rsl_reindexed.index.name = 'TIME'
print(len(cml_rsl_reindexed))  # 207360


# 滤除异常值，即将绝对值超过50的值替换为空值
cml_rsl_reindexed['RSL'] = [np.nan if abs(x) > 50 else x for x in cml_rsl_reindexed['RSL']]
# 线性插值
cml_rsl_reindexed_interpolated = cml_rsl_reindexed.interpolate(method='linear', limit_direction='both')
# 将index列变为普通行
cml_rsl_reindexed_interpolated = cml_rsl_reindexed_interpolated.reset_index()
# print(cml_rsl_reindexed_interpolated)  # 207360
cml_rsl_reindexed_interpolated.to_excel('../generate_file/cml_rsl_processed.xlsx', index=False)

# 检查是否有空缺值
# 方法1: 使用any()
has_nan_any = cml_rsl_reindexed_interpolated.RSL.isna().any()
print("使用any()检查NaN:", has_nan_any)   # False


plt.plot(cml_rsl_reindexed_interpolated.TIME, cml_rsl_reindexed_interpolated.RSL)
plt.show()
