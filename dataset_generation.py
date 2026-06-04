from physics_system import ChuaCircuit, DoublePendulum
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import time

start_time=time.time()
# double pendulum-- dataset generation, 10k configurations
dp=DoublePendulum()
Dataset_DoublePendulum=[]
for i in range(3000):
    sample_parameters=dp.sample_parameters()
    p_groundtruth,T,y_nonoise=dp.solve_system()
    y_noisy=dp.noisy_data_output(y_nonoise)
    print(y_noisy.shape)

    dt=(T[-1]-T[0])/len(T)
    

    estimated_angular_velocity_pendulum1=savgol_filter(x=y_noisy[0,:],deriv=1,delta=dt,polyorder=3,window_length=51)
    estimated_angular_velocity_pendulum2=savgol_filter(x=y_noisy[1,:],deriv=1,delta=dt,polyorder=3,window_length=51)

    df=pd.DataFrame({
        "config_id":i,
        "mass pendulum 1":p_groundtruth["m1"],
        "mass pendulum 2":p_groundtruth["m2"],
        "length pendulum 1":p_groundtruth["l1"],
        "length pendulum 2":p_groundtruth["l2"],
        "time": T,
        "noisy pendulum angle 1":y_noisy[0,:].T,
        "noisy pendulum angle 2":y_noisy[1,:].T,
        "estimated angular velocity pendulum 1": estimated_angular_velocity_pendulum1.T,
        "estimated angular velocity pendulum 2": estimated_angular_velocity_pendulum2.T,
        "no noise angle pendulum 1":y_nonoise[0,:].T,
        "no noise angle pendulum 2":y_nonoise[2,:].T,
        "no noise angularvel pendulum 1":y_nonoise[1,:].T,
        "no noise angularvel pendulum 2":y_nonoise[3,:].T
        })
    Dataset_DoublePendulum.append(df)
    print(f'Config Id: {i} completed for DP')

Dataset=pd.concat(Dataset_DoublePendulum)
Dataset.to_parquet('datasets/dataset_doublependulumpts_setB.parquet',index=False)

totaltime=time.time()-start_time
print(f'time in seconds:{totaltime}')    


# cc=ChuaCircuit()
# Dataset_ChuaCircuit=pd.DataFrame({})
# for j in range(200):
#     sample_parameters=cc.sample_parameters()
#     p_groundtruth_cc,T_cc,y_nonoise_cc=cc.solve_system()
#     y_noisy_cc=cc.noisy_data_output(y_nonoise_cc)
#     # print(y_noisy.shape)

#     # dt=(T[-1]-T[0])/len(T)
    

#     # estimated_angular_velocity_pendulum1=savgol_filter(x=y_noisy[0,:],deriv=1,delta=dt,polyorder=3,window_length=51)
#     # estimated_angular_velocity_pendulum2=savgol_filter(x=y_noisy[1,:],deriv=1,delta=dt,polyorder=3,window_length=51)

#     df_cc=pd.DataFrame({
#         "config_id":j,
#         "inductance":p_groundtruth_cc["L"],
#         "capacitance 1":p_groundtruth_cc["C1"],
#         "capaciatnce 2":p_groundtruth_cc["C2"],
#         "resistance":p_groundtruth_cc["R"],
#         "time": T_cc,
#         "noisy x":y_noisy_cc[0,:].T,
#         "noisy y":y_noisy_cc[1,:].T,
#         "noisy z":y_noisy_cc[2,:].T,
#         "no noise x":y_nonoise_cc[0,:].T,
#         "no noise y":y_nonoise_cc[1,:].T,
#         "no noise z":y_nonoise_cc[2,:].T,
#         })
#     Dataset_ChuaCircuit=pd.concat([Dataset_ChuaCircuit, df_cc])
#     print(f'Config Id: {j} completed for CC')

# Dataset_ChuaCircuit.to_csv('dataset_chuacircuit.csv',index=False)

# totaltime=time.time()-start_time
# print(f'time in seconds:{totaltime}')    
