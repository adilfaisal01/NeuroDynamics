# import pandas as pd
# import matplotlib.pyplot as plt

# xx=pd.read_parquet("datasets/dataset_doublependulumpts.parquet")
# print(xx.columns)
# print(xx[xx['config_id']==99]['time'].size)
# plt.plot(xx[xx['config_id']==99]['time'],xx[xx['config_id']==99]['noisy pendulum angle 1'], label='1')
# plt.plot(xx[xx['config_id']==99]['time'],xx[xx['config_id']==99]['no noise angle pendulum 1'],label='2')
# plt.legend()
# plt.show()

from _hamil