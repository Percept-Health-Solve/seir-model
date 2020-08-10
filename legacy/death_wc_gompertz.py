import pandas as pd 
import numpy as np
import datetime

df_deaths = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv',
        parse_dates=['date'],
        date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
)

df_deaths = df_deaths[['date','WC']]
df_deaths['date'] = [x.date() for x in df_deaths['date']]
df_deaths['Day'] = [(x - datetime.date(2020,3,26)).days for x in df_deaths['date']]
df_deaths['Daily'] = df_deaths['WC'].diff()

from scipy.optimize import curve_fit
def gompertz(x,a,b,c):
    return (a*np.exp(-b*np.exp(-c*x)))

params, pcov = curve_fit(gompertz, xdata=df_deaths['Day'], ydata=df_deaths['WC'], p0=[20000,1,0.2])

df_deaths['Gompertz'] = gompertz(df_deaths['Day'],params[0],params[1],params[2])

df_deaths_append = df_deaths.iloc[0:0].copy()
df_deaths_append['Day'] = list(range(df_deaths['Day'].max()+1,300))
df_deaths_append['Gompertz'] = gompertz(df_deaths_append['Day'],params[0],params[1],params[2])

df_deaths_past = df_deaths.copy()

df_deaths = df_deaths.append(df_deaths_append)
df_deaths['Daily'] = df_deaths['WC'].diff()
df_deaths['Daily_Gompertz'] = df_deaths['Gompertz'].diff()


fig,ax = plt.subplots(3,1,figsize=(20,15))

ax[0].plot(df_deaths_past['Day'],df_deaths_past['WC'])
ax[0].plot(df_deaths_past['Day'],df_deaths_past['Gompertz'],c='green',linestyle='dashed')
ax[0].set_title('Cumulative deaths to date')

ax[1].plot(df_deaths['Day'],df_deaths['WC'])
ax[1].plot(df_deaths['Day'],df_deaths['Gompertz'],c='green',linestyle='dashed')
ax[1].set_title('Cumulative deaths')

ax[2].plot(df_deaths['Day'],df_deaths['Daily'])
ax[2].plot(df_deaths['Day'],df_deaths['Daily_Gompertz'],c='green',linestyle='dashed')
ax[2].set_title('Daily deaths')

fig.tight_layout()

fig.savefig('data/WC_deaths_Gompertz.png')





# drop last observation
df_deaths_past = df_deaths_past[df_deaths_past['Day']<81]
params, pcov = curve_fit(gompertz, xdata=df_deaths_past['Day'], ydata=df_deaths_past['WC'], p0=params)
