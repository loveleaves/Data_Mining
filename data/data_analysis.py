import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小

def plot_plot(x,y,is_mis_val=False):
    if is_mis_val:
        plt.plot(x, y, '-p', color='grey',
                 marker='o',
                 markersize=8, linewidth=2,
                 markerfacecolor='red',
                 markeredgecolor='grey',
                 markeredgewidth=2)
    else:
        plt.plot(x,y)

def data_analysis(data,mis_val,ignore_col=['ID']):
    data_columns = list(data.columns)
    if ignore_col:
        for col_name in ignore_col:
            data_columns.remove(col_name)
    for col_name in data_columns:
        for index in range(data.shape[0]):
            # str_data = str(data[col_name].iat[index])
            # if str_data in mis_val:
            #     plot_plot(index,data[col_name].iat[index])
            plot_plot(index,index)
        plt.show()
