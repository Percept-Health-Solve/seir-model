import numpy as np 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt

deaths = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv',
        parse_dates=['date'],
        date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
    )
deaths = deaths[['date','total']]
deaths.index = deaths['date']
deaths.drop(columns=['date'],inplace=True)
deaths = deaths.resample('1D').ffill().reset_index()

deaths['t'] = [x.days for x in (max(deaths['date']) - deaths['date'])]
t_max = max(deaths['t'])

fig, axes = plt.subplots(5,2,figsize=(20,20))

for i,d in enumerate([1,2,3,4,5,6,7,10,12,14]):
    ax = axes[int(i/2),i-2*int(i/2)]
    tt = np.arange(0,t_max+1,d)
    periodic_deaths = pd.Series(([0] + list(deaths.loc[deaths['t'].isin(tt),'total']))).diff()[1:]
    ax.plot(periodic_deaths)
    ax.set_title(f'Deaths over {d}-day intervals')

plt.show()
fig.savefig('./data/Death_grouping.png')