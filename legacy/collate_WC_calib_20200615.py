import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

for x in range(10):
    filename = 'data/0' + str(x) + '_WC_calib_20200614_prediction.csv'
    pred = pd.read_csv(filename)[['Date','Hospitalised_mean','ICU_mean','Deaths_mean']]
    if x==0:
        pred_sum = pred.copy()
    else:
        pred_sum = pred_sum.merge(pred,on='Date',suffixes=['_'+str(x-1),'_'+str(x)])


for i in ['Hospitalised','ICU','Deaths']:
    cols = [i + '_mean_' + str(x) for x in range(10)]
    pred_sum[i + '_mean_all'] = pred_sum[cols].mean(axis=1)

fig,ax = plt.subplots(1,3,figsize=(20,10))

for x in range(10):
    ax[0].plot(pred_sum['Date'],pred_sum['Hospitalised_mean_' + str(x)])
    ax[1].plot(pred_sum['Date'],pred_sum['ICU_mean_' + str(x)])
    ax[2].plot(pred_sum['Date'],pred_sum['Deaths_mean_' + str(x)])

fig,ax = plt.subplots(1,3,figsize=(20,10))

ax[0].plot(pred_sum['Date'],pred_sum['Hospitalised_mean_all'])
ax[0].set_title('Hospital beds (excl. ICU)')
ax[1].plot(pred_sum['Date'],pred_sum['ICU_mean_all'])
ax[1].set_title('ICU beds')
ax[2].plot(pred_sum['Date'],pred_sum['Deaths_mean_all'])
ax[2].set_title('Deaths')
for i,axis in enumerate(ax):
    axis.set_xticks(range(0,360,30))
    for tick in axis.get_xticklabels():
        tick.set_rotation(45)
fig.suptitle('Western Cape calibration (mean) 15 June 2020- for review, not to be used')
fig.savefig('data/WC_calib_mean_projections_20200615.png')