import numpy as np 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt 


df = pd.read_excel('data/202005 WCDoH Covid19 admissions data.xlsx',sheet_name='WC Covid19 Admissions')
for col in ['date_of_diagnosis','Admission_date','discharge_date','Date_of_ICU_admission']:
  df[col] = [x.date() for x in df[col]]

def hosp_indic(row,hosp_date):
  icu = 0
  hosp = 0
  if row['Admitted_to_ICU'] == 'Yes' and hosp_date >= row['Date_of_ICU_admission']and (pd.isna(row['discharge_date']) or row['discharge_date'] >= hosp_date):
    icu = 1
  if hosp_date >= row['Admission_date'] and (pd.isna(row['discharge_date']) or row['discharge_date'] >= hosp_date):
    hosp = 1 - icu
  return [hosp,icu]

first_date = df.loc[~pd.isna(df['Admission_date']),'Admission_date'].min()
last_date = df.loc[~pd.isna(df['discharge_date']),'discharge_date'].max()
timespan = divmod((last_date - first_date).days, 1)[0] + 1

hospitalisations = pd.DataFrame({'Date':[first_date + datetime.timedelta(days=x) for x in range(timespan)],
                                 'Hospital_excl_ICU':0,
                                 'ICU':0})

for hosp_date in hospitalisations['Date']:
  out = np.asarray(list(df.apply(hosp_indic,axis=1,hosp_date=hosp_date))).sum(axis=0)
  hospitalisations.loc[hospitalisations['Date']==hosp_date,'Hospital_excl_ICU'] = out[0]
  hospitalisations.loc[hospitalisations['Date']==hosp_date,'ICU'] = out[1]

hospitalisations.to_csv('data/WC_hosp_ICU.csv',index=False)


from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time


def durn(start,end,max):
  if pd.isna(start):
    return(-1)
  else:
    if pd.isna(end):
      end = max
    return (divmod((end - start).days, 1)[0])

n = df.shape[0]
durn_hosp_to_discharge = [0] * n
observed_hosp_to_discharge = [False] * n
durn_hosp_to_death = [0] * n
observed_hosp_to_death = [False] * n
durn_hosp_to_icu = [0] * n
observed_hosp_to_icu = [False] * n
durn_icu_to_discharge = [0] * n
observed_icu_to_discharge = [False] * n
durn_icu_to_death = [0] * n
observed_icu_to_death = [False] * n

for i in range(n):
  if df.at[i,'admission_status'] == 'Died':
    durn_hosp_to_discharge[i] = -1
    durn_icu_to_discharge[i] = -1
    if df.at[i,'Admitted_to_ICU'] == 'No':
      durn_hosp_to_death[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
      durn_hosp_to_icu[i] = -1
      if ~pd.isna(df.at[i,'discharge_date']):
        observed_hosp_to_death[i] = True
      else:
        print(f'Error in row {i}: death without discharge date')
      durn_icu_to_death[i] = -1
    else:
      durn_hosp_to_death[i] = -1
      durn_hosp_to_icu[i] = durn(df.at[i,'Admission_date'],df.at[i,'Date_of_ICU_admission'],last_date)
      observed_hosp_to_icu[i] = True
      durn_icu_to_death[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
      if ~pd.isna(df.at[i,'discharge_date']):
        observed_icu_to_death[i] = True
      else:
        print(f'Error in row {i}: death without discharge date')
  else:
    durn_hosp_to_death[i] = -1
    durn_icu_to_death[i] = -1
    if df.at[i,'Admitted_to_ICU'] == 'Yes':
      durn_hosp_to_discharge[i] = -1
      durn_hosp_to_icu[i] = durn(df.at[i,'Admission_date'],df.at[i,'Date_of_ICU_admission'],last_date)
      observed_hosp_to_icu[i] = True
      durn_icu_to_discharge[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
      if not(pd.isna(df.at[i,'discharge_date'])):
        observed_icu_to_discharge[i] = True
    else:
      durn_hosp_to_icu[i] = -1
      durn_icu_to_discharge[i] = -1
      durn_hosp_to_discharge[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
      if not(pd.isna(df.at[i,'discharge_date'])):
        observed_hosp_to_discharge[i] = True

df['durn_hosp_to_discharge']  = durn_hosp_to_discharge
df['observed_hosp_to_discharge'] = observed_hosp_to_discharge
df['durn_hosp_to_death'] = durn_hosp_to_death
df['observed_hosp_to_death'] = observed_hosp_to_death
df['durn_hosp_to_icu'] = durn_hosp_to_icu
df['observed_hosp_to_icu'] = observed_hosp_to_icu
df['durn_icu_to_discharge'] = durn_icu_to_discharge
df['observed_icu_to_discharge'] = observed_icu_to_discharge
df['durn_icu_to_death'] = durn_icu_to_death
df['observed_icu_to_death'] = observed_icu_to_death

df.to_csv('data/durations.csv')
# TODO: logic above is faulty, assigns right-censored cases only to discharge, but not clear how best to correct

# Kaplan Meier estimators
kmf_hosp_to_discharge = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_discharge'] != -1,'durn_hosp_to_discharge'].values,
                                event_observed = df.loc[df['durn_hosp_to_discharge'] != -1,'observed_hosp_to_discharge'].values)
kmf_hosp_to_death = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_death'] != -1,'durn_hosp_to_death'].values,
                            event_observed = df.loc[df['durn_hosp_to_death'] != -1,'observed_hosp_to_death'].values)
kmf_hosp_to_icu = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_icu'] != -1,'durn_hosp_to_icu'].values,
                            event_observed = df.loc[df['durn_hosp_to_icu'] != -1,'observed_hosp_to_icu'].values)
kmf_icu_to_discharge = KaplanMeierFitter().fit(durations = df.loc[df['durn_icu_to_discharge'] != -1,'durn_icu_to_discharge'].values,
                               event_observed = df.loc[df['durn_icu_to_discharge'] != -1,'observed_icu_to_discharge'].values)
kmf_icu_to_death = KaplanMeierFitter().fit(durations = df.loc[df['durn_icu_to_death'] != -1,'durn_icu_to_death'].values,
                           event_observed = df.loc[df['durn_icu_to_death'] != -1,'observed_icu_to_death'].values)

n_hosp_to_discharge = df[df['durn_hosp_to_discharge'] != -1].shape[0]
n_hosp_to_death = df[df['durn_hosp_to_death'] != -1].shape[0]
n_hosp_to_icu = df[df['durn_hosp_to_icu'] != -1].shape[0]
n_icu_to_discharge = df[df['durn_icu_to_discharge'] != -1].shape[0]
n_icu_to_death = df[df['durn_icu_to_death'] != -1].shape[0]

mean_hosp_to_discharge = restricted_mean_survival_time(kmf_hosp_to_discharge,50).mean()
mean_hosp_to_death = restricted_mean_survival_time(kmf_hosp_to_death,50).mean()
mean_hosp_to_icu = restricted_mean_survival_time(kmf_hosp_to_icu,50).mean()
mean_icu_to_discharge = restricted_mean_survival_time(kmf_icu_to_discharge,50).mean()
mean_icu_to_death = restricted_mean_survival_time(kmf_icu_to_death,50).mean()

# plot
fig, axes = plt.subplots(5,1,figsize=(20,20))

kmf_hosp_to_discharge.plot(ax=axes[0])
kmf_hosp_to_death.plot(ax=axes[1])
kmf_hosp_to_icu.plot(ax=axes[2])
kmf_icu_to_discharge.plot(ax=axes[3])
kmf_icu_to_death.plot(ax=axes[4])

axes[0].set_title(f'Hospital to discharge for non-ICU cases (allowing for right-censoring), n = {n_hosp_to_discharge}')
axes[1].set_title(f'Hospital to death for non-ICU cases (not allowing for right-censoring), n = {n_hosp_to_death}')
axes[2].set_title(f'Hospital to ICU (not allowing for right-censoring), n = {n_hosp_to_icu}')
axes[3].set_title(f'ICU to discharge (allowing for right-censoring), n = {n_icu_to_discharge}')
axes[4].set_title(f'ICU to death (not allowing for right-censoring), n = {n_icu_to_death}')

axes[0].axvline(kmf_hosp_to_discharge.median_survival_time_,linestyle='--',color='red')
axes[1].axvline(kmf_hosp_to_death.median_survival_time_,linestyle='--',color='red')
axes[2].axvline(kmf_hosp_to_icu.median_survival_time_,linestyle='--',color='red')
axes[3].axvline(kmf_icu_to_discharge.median_survival_time_,linestyle='--',color='red')
axes[4].axvline(kmf_icu_to_death.median_survival_time_,linestyle='--',color='red')

axes[0].axvline(mean_hosp_to_discharge,linestyle='--',color='green')
axes[1].axvline(mean_hosp_to_death,linestyle='--',color='green')
axes[2].axvline(mean_hosp_to_icu,linestyle='--',color='green')
axes[3].axvline(mean_icu_to_discharge,linestyle='--',color='green')
axes[4].axvline(mean_icu_to_death,linestyle='--',color='green')


from matplotlib.lines import Line2D
v_lines = [Line2D([0], [0], color='blue', lw=1),
           Line2D([0], [0], color='red', linestyle = '--', lw=1),
           Line2D([0], [0], color='green', lw=1)]

for i in range(5):
  axes[i].set_xlim(-1,30)
  axes[i].set_xticks(list(range(31)))
  axes[i].set_xticklabels(list(range(31)))
  axes[i].legend(v_lines,['KM estimate','Median','Mean'])

plt.tight_layout()
fig.savefig('data/KM_estimates.png')
plt.show()

