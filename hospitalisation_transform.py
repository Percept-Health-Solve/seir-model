import numpy as np 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import argparse

# survival analysis
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--age_groups', action='store_true')

# read data
df = pd.read_csv('data/202005 WCDoH Covid19 admissions data v3.csv',
                 parse_dates=['date_of_diagnosis','Admission_date','discharge_date','Date_of_ICU_admission'])
for col in ['date_of_diagnosis','Admission_date','discharge_date','Date_of_ICU_admission']:
    df[col] = [x.date() for x in df[col]]

def hosp_indic(row,hosp_date):
    icu = 0
    hosp = 0
    if row['Admitted_to_ICU'] == 'Yes' and hosp_date >= row['Date_of_ICU_admission'] and (pd.isna(row['discharge_date']) or row['discharge_date'] >= hosp_date):
        icu = 1
    if hosp_date >= row['Admission_date'] and (pd.isna(row['discharge_date']) or row['discharge_date'] >= hosp_date):
        hosp = 1 - icu
    return [hosp,icu]

# get start and end dates of data
first_date = df.loc[~pd.isna(df['Admission_date']),'Admission_date'].min()
last_date = df.loc[~pd.isna(df['discharge_date']),'discharge_date'].max()
timespan = divmod((last_date - first_date).days, 1)[0] + 1

# set up output dataframe
hospitalisations = pd.DataFrame({'Date':[first_date + datetime.timedelta(days=x) for x in range(timespan)],
                                 'Hospital_excl_ICU':0,
                                 'ICU':0})

# populate
for hosp_date in hospitalisations['Date']:
    out = np.asarray(list(df.apply(hosp_indic,axis=1,hosp_date=hosp_date))).sum(axis=0)
    hospitalisations.loc[hospitalisations['Date']==hosp_date,'Hospital_excl_ICU'] = out[0]
    hospitalisations.loc[hospitalisations['Date']==hosp_date,'ICU'] = out[1]

# write to CSV
hospitalisations.to_csv('data/WC_hosp_ICU.csv',index=False)


def durn(start,end,max):
    '''
    Calculate duration in days from start to end (if defined) or max (if end not defined)
    '''
    if pd.isna(start):
      return(-1)
    else:
      if pd.isna(end):
        end = max
      return (divmod((end - start).days, 1)[0])


def km_estimate(df, fig_path='data/KM_estimates.png', csv_path='data/durations.csv'):
    # calculate observed mortality rates for weighting righ-censored data
    death_rate_hosp = df[(df['admission_status'] == 'Died') & (df['Admitted_to_ICU'] == 'No')].shape[0] / \
                      df[(df['admission_status'] != 'Inpatient') & (df['Admitted_to_ICU'] == 'No')].shape[0]
    death_rate_icu = df[(df['admission_status'] == 'Died') & (df['Admitted_to_ICU'] == 'Yes')].shape[0] / \
                     df[(df['admission_status'] != 'Inpatient') & (df['Admitted_to_ICU'] == 'Yes')].shape[0]

    # initialise duration, observed (Boolean) and weight lists
    n = df.shape[0]
    durn_hosp_to_discharge = [0] * n
    observed_hosp_to_discharge = [False] * n
    weight_hosp_to_discharge = [1] * n
    durn_hosp_to_death = [0] * n
    observed_hosp_to_death = [False] * n
    weight_hosp_to_death = [1] * n
    durn_hosp_to_icu = [0] * n
    observed_hosp_to_icu = [False] * n
    weight_hosp_to_icu = [1] * n
    durn_icu_to_discharge = [0] * n
    observed_icu_to_discharge = [False] * n
    weight_icu_to_discharge = [1] * n
    durn_icu_to_death = [0] * n
    observed_icu_to_death = [False] * n
    weight_icu_to_death = [1] * n

    # data checks
    death_wo_discharge_date = sum((df['admission_status'] == 'Died') & (pd.isna(df['discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status'] == 'Discharged') & (pd.isna(df['discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status'] == 'Inpatient') & (~pd.isna(df['discharge_date'])))
    icu_wo_icu_admission = sum((df['Admitted_to_ICU'] == 'Yes') & (pd.isna(df['Date_of_ICU_admission'])))
    print(f'Deaths without discharge dates: {death_wo_discharge_date}')
    print(f'Discharges without discharge dates: {discharge_wo_discharge_date}')
    print(f'Inpatients with discharge dates: {discharge_wo_discharge_date}')
    print(f'ICU without ICU admission dates: {icu_wo_icu_admission}')

    # assign survival times, observation flags and weight
    # TODO: make this more Pythonic!
    for i in range(n):
        if df.at[i, 'admission_status'] == 'Died':
            durn_hosp_to_discharge[i] = -1
            durn_icu_to_discharge[i] = -1
            if df.at[i, 'Admitted_to_ICU'] == 'No':
                durn_hosp_to_death[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'discharge_date'], last_date)
                durn_hosp_to_icu[i] = -1
                observed_hosp_to_death[i] = True
                durn_icu_to_death[i] = -1
            else:
                durn_hosp_to_death[i] = -1
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_death[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'discharge_date'], last_date)
                observed_icu_to_death[i] = True
        elif df.at[i, 'admission_status'] == 'Discharged':
            durn_hosp_to_death[i] = -1
            durn_icu_to_death[i] = -1
            if df.at[i, 'Admitted_to_ICU'] == 'Yes':
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'discharge_date'],
                                                last_date)
                observed_icu_to_discharge[i] = True
            else:
                durn_hosp_to_icu[i] = -1
                durn_icu_to_discharge[i] = -1
                durn_hosp_to_discharge[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'discharge_date'], last_date)
                observed_hosp_to_discharge[i] = True
        else:  # inpatients
            if df.at[i, 'Admitted_to_ICU'] == 'No':
                durn_hosp_to_icu[i] = -1  # assume none, given that most ICU cases are admitted directly
                durn_hosp_to_discharge[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'discharge_date'], last_date)
                durn_hosp_to_death[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'discharge_date'], last_date)
                weight_hosp_to_discharge[i] = 1 - death_rate_hosp
                weight_hosp_to_death[i] = death_rate_hosp
            else:
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'discharge_date'],
                                                last_date)
                durn_icu_to_death[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'discharge_date'], last_date)
                weight_icu_to_discharge[i] = 1 - death_rate_icu
                weight_icu_to_death[i] = death_rate_icu

    # populate dataframe from lists
    df['durn_hosp_to_discharge'] = durn_hosp_to_discharge
    df['observed_hosp_to_discharge'] = observed_hosp_to_discharge
    df['weight_hosp_to_discharge'] = weight_hosp_to_discharge
    df['durn_hosp_to_death'] = durn_hosp_to_death
    df['observed_hosp_to_death'] = observed_hosp_to_death
    df['weight_hosp_to_death'] = weight_hosp_to_death
    df['durn_hosp_to_icu'] = durn_hosp_to_icu
    df['observed_hosp_to_icu'] = observed_hosp_to_icu
    df['weight_hosp_to_icu'] = weight_hosp_to_icu
    df['durn_icu_to_discharge'] = durn_icu_to_discharge
    df['observed_icu_to_discharge'] = observed_icu_to_discharge
    df['weight_icu_to_discharge'] = weight_icu_to_discharge
    df['durn_icu_to_death'] = durn_icu_to_death
    df['observed_icu_to_death'] = observed_icu_to_death
    df['weight_icu_to_death'] = weight_icu_to_death

    # write duration data to CSV
    df.to_csv(csv_path)

    # Kaplan Meier estimates
    kmf_hosp_to_discharge = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_hosp_to_discharge'] != -1, 'durn_hosp_to_discharge'].values,
        event_observed=df.loc[df['durn_hosp_to_discharge'] != -1, 'observed_hosp_to_discharge'].values,
        weights=df.loc[df['durn_hosp_to_discharge'] != -1, 'weight_hosp_to_discharge'].values)
    kmf_hosp_to_death = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_hosp_to_death'] != -1, 'durn_hosp_to_death'].values,
        event_observed=df.loc[df['durn_hosp_to_death'] != -1, 'observed_hosp_to_death'].values,
        weights=df.loc[df['durn_hosp_to_death'] != -1, 'weight_hosp_to_death'].values)
    kmf_hosp_to_icu = KaplanMeierFitter().fit(durations=df.loc[df['durn_hosp_to_icu'] != -1, 'durn_hosp_to_icu'].values,
                                              event_observed=df.loc[
                                                  df['durn_hosp_to_icu'] != -1, 'observed_hosp_to_icu'].values,
                                              weights=df.loc[df['durn_hosp_to_icu'] != -1, 'weight_hosp_to_icu'].values)
    kmf_icu_to_discharge = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_icu_to_discharge'] != -1, 'durn_icu_to_discharge'].values,
        event_observed=df.loc[df['durn_icu_to_discharge'] != -1, 'observed_icu_to_discharge'].values,
        weights=df.loc[df['durn_icu_to_discharge'] != -1, 'weight_icu_to_discharge'].values)
    kmf_icu_to_death = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_icu_to_death'] != -1, 'durn_icu_to_death'].values,
        event_observed=df.loc[df['durn_icu_to_death'] != -1, 'observed_icu_to_death'].values,
        weights=df.loc[df['durn_icu_to_death'] != -1, 'weight_icu_to_death'].values)

    n_hosp_to_discharge = sum((df['durn_hosp_to_discharge'] != -1) & (df['durn_hosp_to_icu'] == -1))
    n_hosp_to_discharge_censored = sum(df['weight_hosp_to_discharge'].between(0.01, 0.99))
    n_hosp_to_death = sum((df['durn_hosp_to_death'] != -1) & (df['durn_hosp_to_icu'] == -1))
    n_hosp_to_death_censored = sum(df['weight_hosp_to_discharge'].between(0.01, 0.99))
    n_hosp_to_icu = sum(df['durn_hosp_to_icu'] != -1)
    n_hosp_to_icu_censored = sum(df['weight_hosp_to_icu'].between(0.01, 0.99))
    n_icu_to_discharge = sum((df['durn_icu_to_discharge'] != -1) & (df['durn_hosp_to_icu'] != -1))
    n_icu_to_discharge_censored = sum(df['weight_icu_to_discharge'].between(0.01, 0.99))
    n_icu_to_death = sum((df['durn_icu_to_death'] != -1) & (df['durn_hosp_to_icu'] != -1))
    n_icu_to_death_censored = sum(df['weight_icu_to_death'].between(0.01, 0.99))

    mean_hosp_to_discharge = restricted_mean_survival_time(kmf_hosp_to_discharge, 50).mean()
    mean_hosp_to_death = restricted_mean_survival_time(kmf_hosp_to_death, 50).mean()
    mean_hosp_to_icu = restricted_mean_survival_time(kmf_hosp_to_icu, 50).mean()
    mean_icu_to_discharge = restricted_mean_survival_time(kmf_icu_to_discharge, 50).mean()
    mean_icu_to_death = restricted_mean_survival_time(kmf_icu_to_death, 50).mean()

    # plot
    fig, axes = plt.subplots(5, 1, figsize=(20, 20))

    kmf_hosp_to_discharge.plot(ax=axes[0])
    kmf_hosp_to_death.plot(ax=axes[1])
    kmf_hosp_to_icu.plot(ax=axes[2])
    kmf_icu_to_discharge.plot(ax=axes[3])
    kmf_icu_to_death.plot(ax=axes[4])

    axes[0].set_title(
        f'Hospital to discharge for non-ICU cases: n = {n_hosp_to_discharge} including {n_hosp_to_discharge_censored} right-censored at weight {1 - death_rate_hosp:.3f}')
    axes[1].set_title(
        f'Hospital to death for non-ICU cases: n = {n_hosp_to_death} including {n_hosp_to_death_censored} right-censored at weight {death_rate_hosp:.3f}')
    axes[2].set_title(f'Hospital to ICU: n = {n_hosp_to_icu} including {n_hosp_to_icu_censored} right-censored ')
    axes[3].set_title(
        f'ICU to discharge: n = {n_icu_to_discharge} including {n_icu_to_discharge_censored} right-censored at weight {1 - death_rate_icu:.3f}')
    axes[4].set_title(
        f'ICU to death: n = {n_icu_to_death} including {n_icu_to_death_censored} right-censored at weight {death_rate_icu:.3f}')

    axes[0].axvline(kmf_hosp_to_discharge.median_survival_time_, linestyle='--', color='red')
    axes[1].axvline(kmf_hosp_to_death.median_survival_time_, linestyle='--', color='red')
    axes[2].axvline(kmf_hosp_to_icu.median_survival_time_, linestyle='--', color='red')
    axes[3].axvline(kmf_icu_to_discharge.median_survival_time_, linestyle='--', color='red')
    axes[4].axvline(kmf_icu_to_death.median_survival_time_, linestyle='--', color='red')

    axes[0].axvline(mean_hosp_to_discharge, linestyle='--', color='green')
    axes[1].axvline(mean_hosp_to_death, linestyle='--', color='green')
    axes[2].axvline(mean_hosp_to_icu, linestyle='--', color='green')
    axes[3].axvline(mean_icu_to_discharge, linestyle='--', color='green')
    axes[4].axvline(mean_icu_to_death, linestyle='--', color='green')

    v_lines = [Line2D([0], [0], color='blue', lw=1),
               Line2D([0], [0], color='red', linestyle='--', lw=1),
               Line2D([0], [0], color='green', lw=1)]

    for i in range(5):
        axes[i].set_xlim(-1, 30)
        axes[i].set_xticks(list(range(31)))
        axes[i].set_xticklabels(list(range(31)))
        axes[i].legend(v_lines, ['KM estimate', 'Median', 'Mean'])

    plt.tight_layout()
    fig.savefig(fig_path)
    # plt.show()# calculate observed mortality rates for weighting righ-censored data

    death_rate_hosp = df[(df['admission_status']=='Died') & (df['Admitted_to_ICU']=='No')].shape[0] / \
                        df[(df['admission_status']!='Inpatient') & (df['Admitted_to_ICU']=='No')].shape[0]
    death_rate_icu = df[(df['admission_status']=='Died') & (df['Admitted_to_ICU']=='Yes')].shape[0] / \
                        df[(df['admission_status']!='Inpatient') & (df['Admitted_to_ICU']=='Yes')].shape[0]

    # initialise duration, observed (Boolean) and weight lists
    n = df.shape[0]
    durn_hosp_to_discharge = [0] * n
    observed_hosp_to_discharge = [False] * n
    weight_hosp_to_discharge = [1] * n
    durn_hosp_to_death = [0] * n
    observed_hosp_to_death = [False] * n
    weight_hosp_to_death = [1] * n
    durn_hosp_to_icu = [0] * n
    observed_hosp_to_icu = [False] * n
    weight_hosp_to_icu = [1] * n
    durn_icu_to_discharge = [0] * n
    observed_icu_to_discharge = [False] * n
    weight_icu_to_discharge = [1] * n
    durn_icu_to_death = [0] * n
    observed_icu_to_death = [False] * n
    weight_icu_to_death = [1] * n

    # data checks
    death_wo_discharge_date = sum((df['admission_status']=='Died') & (pd.isna(df['discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status']=='Discharged') & (pd.isna(df['discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status']=='Inpatient') & (~pd.isna(df['discharge_date'])))
    icu_wo_icu_admission = sum((df['Admitted_to_ICU']=='Yes') & (pd.isna(df['Date_of_ICU_admission'])))
    print(f'Deaths without discharge dates: {death_wo_discharge_date}')
    print(f'Discharges without discharge dates: {discharge_wo_discharge_date}')
    print(f'Inpatients with discharge dates: {discharge_wo_discharge_date}')
    print(f'ICU without ICU admission dates: {icu_wo_icu_admission}')

    # assign survival times, observation flags and weight
    # TODO: make this more Pythonic!
    for i in range(n):
        if df.at[i,'admission_status'] == 'Died':
            durn_hosp_to_discharge[i] = -1
            durn_icu_to_discharge[i] = -1
            if df.at[i,'Admitted_to_ICU'] == 'No':
                durn_hosp_to_death[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
                durn_hosp_to_icu[i] = -1
                observed_hosp_to_death[i] = True
                durn_icu_to_death[i] = -1
            else:
                durn_hosp_to_death[i] = -1
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i,'Admission_date'],df.at[i,'Date_of_ICU_admission'],last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_death[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
                observed_icu_to_death[i] = True
        elif df.at[i,'admission_status'] == 'Discharged':
            durn_hosp_to_death[i] = -1
            durn_icu_to_death[i] = -1
            if df.at[i,'Admitted_to_ICU'] == 'Yes':
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i,'Admission_date'],df.at[i,'Date_of_ICU_admission'],last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
                observed_icu_to_discharge[i] = True
            else:
                durn_hosp_to_icu[i] = -1
                durn_icu_to_discharge[i] = -1
                durn_hosp_to_discharge[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
                observed_hosp_to_discharge[i] = True
        else: # inpatients
            if df.at[i,'Admitted_to_ICU'] == 'No':
                durn_hosp_to_icu[i] = -1  # assume none, given that most ICU cases are admitted directly
                durn_hosp_to_discharge[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
                durn_hosp_to_death[i] = durn(df.at[i,'Admission_date'],df.at[i,'discharge_date'],last_date)
                weight_hosp_to_discharge[i] = 1 - death_rate_hosp
                weight_hosp_to_death[i] = death_rate_hosp
            else:
                durn_hosp_to_icu[i] = durn(df.at[i,'Admission_date'],df.at[i,'Date_of_ICU_admission'],last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
                durn_icu_to_death[i] = durn(df.at[i,'Date_of_ICU_admission'],df.at[i,'discharge_date'],last_date)
                weight_icu_to_discharge[i] = 1 - death_rate_icu
                weight_icu_to_death[i] = death_rate_icu

    # populate dataframe from lists
    df['durn_hosp_to_discharge']  = durn_hosp_to_discharge
    df['observed_hosp_to_discharge'] = observed_hosp_to_discharge
    df['weight_hosp_to_discharge'] = weight_hosp_to_discharge
    df['durn_hosp_to_death'] = durn_hosp_to_death
    df['observed_hosp_to_death'] = observed_hosp_to_death
    df['weight_hosp_to_death'] = weight_hosp_to_death
    df['durn_hosp_to_icu'] = durn_hosp_to_icu
    df['observed_hosp_to_icu'] = observed_hosp_to_icu
    df['weight_hosp_to_icu'] = weight_hosp_to_icu
    df['durn_icu_to_discharge'] = durn_icu_to_discharge
    df['observed_icu_to_discharge'] = observed_icu_to_discharge
    df['weight_icu_to_discharge'] = weight_icu_to_discharge
    df['durn_icu_to_death'] = durn_icu_to_death
    df['observed_icu_to_death'] = observed_icu_to_death
    df['weight_icu_to_death'] = weight_icu_to_death

    # write duration data to CSV
    df.to_csv('data/durations.csv')

    # Kaplan Meier estimates
    kmf_hosp_to_discharge = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_discharge'] != -1,'durn_hosp_to_discharge'].values,
                                                    event_observed = df.loc[df['durn_hosp_to_discharge'] != -1,'observed_hosp_to_discharge'].values,
                                                    weights = df.loc[df['durn_hosp_to_discharge'] != -1,'weight_hosp_to_discharge'].values)
    kmf_hosp_to_death = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_death'] != -1,'durn_hosp_to_death'].values,
                                                event_observed = df.loc[df['durn_hosp_to_death'] != -1,'observed_hosp_to_death'].values,
                                                weights = df.loc[df['durn_hosp_to_death'] != -1,'weight_hosp_to_death'].values)
    kmf_hosp_to_icu = KaplanMeierFitter().fit(durations = df.loc[df['durn_hosp_to_icu'] != -1,'durn_hosp_to_icu'].values,
                                              event_observed = df.loc[df['durn_hosp_to_icu'] != -1,'observed_hosp_to_icu'].values,
                                              weights = df.loc[df['durn_hosp_to_icu'] != -1,'weight_hosp_to_icu'].values)
    kmf_icu_to_discharge = KaplanMeierFitter().fit(durations = df.loc[df['durn_icu_to_discharge'] != -1,'durn_icu_to_discharge'].values,
                                                   event_observed = df.loc[df['durn_icu_to_discharge'] != -1,'observed_icu_to_discharge'].values,
                                                   weights = df.loc[df['durn_icu_to_discharge'] != -1,'weight_icu_to_discharge'].values)
    kmf_icu_to_death = KaplanMeierFitter().fit(durations = df.loc[df['durn_icu_to_death'] != -1,'durn_icu_to_death'].values,
                                               event_observed = df.loc[df['durn_icu_to_death'] != -1,'observed_icu_to_death'].values,
                                               weights = df.loc[df['durn_icu_to_death'] != -1,'weight_icu_to_death'].values)

    n_hosp_to_discharge = sum((df['durn_hosp_to_discharge']!=-1) & (df['durn_hosp_to_icu']==-1))
    n_hosp_to_discharge_censored = sum(df['weight_hosp_to_discharge'].between(0.01,0.99))
    n_hosp_to_death = sum((df['durn_hosp_to_death']!=-1) & (df['durn_hosp_to_icu']==-1))
    n_hosp_to_death_censored = sum(df['weight_hosp_to_discharge'].between(0.01,0.99))
    n_hosp_to_icu = sum(df['durn_hosp_to_icu']!=-1)
    n_hosp_to_icu_censored = sum(df['weight_hosp_to_icu'].between(0.01,0.99))
    n_icu_to_discharge = sum((df['durn_icu_to_discharge']!=-1) & (df['durn_hosp_to_icu']!=-1))
    n_icu_to_discharge_censored = sum(df['weight_icu_to_discharge'].between(0.01,0.99))
    n_icu_to_death = sum((df['durn_icu_to_death']!=-1) & (df['durn_hosp_to_icu']!=-1))
    n_icu_to_death_censored = sum(df['weight_icu_to_death'].between(0.01,0.99))

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

    axes[0].set_title(f'Hospital to discharge for non-ICU cases: n = {n_hosp_to_discharge} including {n_hosp_to_discharge_censored} right-censored at weight {1-death_rate_hosp:.3f}')
    axes[1].set_title(f'Hospital to death for non-ICU cases: n = {n_hosp_to_death} including {n_hosp_to_death_censored} right-censored at weight {death_rate_hosp:.3f}')
    axes[2].set_title(f'Hospital to ICU: n = {n_hosp_to_icu} including {n_hosp_to_icu_censored} right-censored ')
    axes[3].set_title(f'ICU to discharge: n = {n_icu_to_discharge} including {n_icu_to_discharge_censored} right-censored at weight {1-death_rate_icu:.3f}')
    axes[4].set_title(f'ICU to death: n = {n_icu_to_death} including {n_icu_to_death_censored} right-censored at weight {death_rate_icu:.3f}')

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


    v_lines = [Line2D([0], [0], color='blue', lw=1),
               Line2D([0], [0], color='red', linestyle = '--', lw=1),
               Line2D([0], [0], color='green', lw=1)]

    for i in range(5):
        axes[i].set_xlim(-1,30)
        axes[i].set_xticks(list(range(31)))
        axes[i].set_xticklabels(list(range(31)))
        axes[i].legend(v_lines,['KM estimate','Median','Mean'])

    plt.tight_layout()
    fig.savefig(fig_path)
    # plt.show()


args = parser.parse_args()
if args.age_groups:
    print('Doing KM estimate for multiple age groups')
    age_groups = [['0 - 5', '5 - 10'], ['10 - 15', '15 - 20'], ['20 - 25', '25 - 30'], ['30 - 35', '35 - 40'],
                  ['40 - 45', '45 - 50'], ['50 - 55', '55 - 60'], ['60 - 65', '65 - 70'], ['70 - 75', '75 - 80'],
                  ['80 - 85', '85 - 90', '90 - 95', '95 - 100']]
    suffixes = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    for suffix, group in zip(suffixes, age_groups):
        print(f'Calculating KM for {suffix}')
        filter = df['agegroup'] == group[0]
        for i in range(1, len(group)):
            filter = (filter) | (df['agegroup'] == group[i])
        km_estimate(df[filter].reset_index(), fig_path=f'data/KM_estimates_{suffix}.png', csv_path=f'data/durations_{suffix}.csv')
else:
    km_estimate(df)



