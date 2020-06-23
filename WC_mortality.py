import pandas as pd 
import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import datetime
#from scipy.interpolate import BSpline

sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (13, 8)
mpl.rcParams['figure.dpi'] = 100

#df = pd.read_excel('./data/202005 WCDoH Covid19 admissions data v3.xlsx',sheet_name = 'WC Covid 19 Admissions')
df = pd.read_csv('./data/AdmissionsList.csv',
                  parse_dates=['Admission_date','discharge_date','Date_of_ICU_admission'])

# remove records with discharge date pre admission date
df = df[~(df['discharge_date'] < df['Admission_date'])].reset_index()
print(f'Number of records in base file: {df.shape[0]:,}')
print(df['admission_status'].value_counts())

# drop last 5 days of information given reporting delays
max_date = df['Admission_date'].max() - datetime.timedelta(days=5)
df = df[df['Admission_date'] <= max_date]
df.loc[df['discharge_date'] > max_date,'admission_status'] = 'Inpatient'
print(f'Number of records after removing last 5 days: {df.shape[0]:,}')
print(df['admission_status'].value_counts())

# drop no age or inpatients
df = df[df['agegroup'] != 'Not recorded']
print(f'After dropping records with no age recorded: {df.shape[0]:,}')
df = df[df['admission_status'] != 'Inpatient']
print(f'After dropping current inpatients: {df.shape[0]:,}')
print(df['admission_status'].value_counts())

# change death status to died for those who died after discharge
print('Before changing death status:\n')
print(df['admission_status'].value_counts())
df.loc[~pd.isna(df['date_of_death']),'admission_status'] = 'Died'
print('\nAfter changing death status:\n')
print(df['admission_status'].value_counts())

# get total deaths for adjustment
df_deaths = pd.read_csv(
    'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv',
    parse_dates=['date'],
    date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
)
death_total = df_deaths.loc[df_deaths['date']==max_date,'WC'].values[0]
death_hospital = df[df['admission_status']=='Died'].shape[0]
death_adjt = death_total / death_hospital

df['Count'] = 1

def tenyr_ageband(agegroup):
    lower = int((int(agegroup[:2])+2)/10)*10
    if lower >= 80:
        return('80+')
    else:
        return(str(lower) + ' - ' + str(lower+9))

df['10yr_ageband'] = df['agegroup'].apply(tenyr_ageband)

df_h = df[df['Admitted_to_ICU']=='No']
df_c = df[df['Admitted_to_ICU']=='Yes']

mort = pd.pivot_table(df,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
mort['Cases'] = mort['Died'] + mort['Discharged']
mort['Adj_deaths'] = mort['Died'] 
mort['Mid_age'] = [5,15,25,35,45,55,65,76,85]

mort_h = pd.pivot_table(df_h,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
mort_h['Cases'] = mort_h['Died'] + mort_h['Discharged']
mort_h['Adj_deaths'] = mort_h['Died'] 
mort_h['Mid_age'] = [5,15,25,35,45,55,65,76,85]

mort_c = pd.pivot_table(df_c,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
mort_c['Cases'] = mort_c['Died'] + mort_c['Discharged']
mort_c['Adj_deaths'] = mort_c['Died'] 
mort_c['Mid_age'] = [5,15,25,35,45,55,65,76,85]

alpha = 0.05
z = st.norm.ppf(1 - alpha/2)

def conf_int(deaths,n):
    p = deaths/n
    if (n > 15) and (p > 0.1):
        # normal approximation
        sampling_err = np.sqrt(p*(1-p)/n)
        lo = max(0,p - z * sampling_err)
        hi = min(1,p + z * sampling_err)
    else: #elif (n > 15) and (p <= 0.1):
        # Poisson approximation
        if deaths == 0:
            lo = 0
        else:
            lo = st.chi2.ppf(alpha/2, 2*deaths) / (2 * n)
        hi = min(1,st.chi2.ppf(1 - alpha/2, 2*deaths + 2) / (2 * n))
    #else: binomial search, ignored for now
    return lo,p,hi

mort_rates = mort.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases']), axis=1)
mort['p_low'] = [x[0] for x in mort_rates]
mort['p_mean'] = [x[1] for x in mort_rates]
mort['p_high'] = [x[2] for x in mort_rates]
mort['p_scaled'] = mort['p_mean'] * death_adjt
#k = 3
#mort['fitted'] = BSpline(mort['Mid_age'],mort['p_mean'],k)(mort['Mid_age'])

mort_rates_h = mort_h.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases']), axis=1)
mort_h['p_low'] = [x[0] for x in mort_rates_h]
mort_h['p_mean'] = [x[1] for x in mort_rates_h]
mort_h['p_high'] = [x[2] for x in mort_rates_h]
mort_h['p_scaled'] = mort_h['p_mean'] * death_adjt

mort_rates_c = mort_c.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases']), axis=1)
mort_c['p_low'] = [x[0] for x in mort_rates_c]
mort_c['p_mean'] = [x[1] for x in mort_rates_c]
mort_c['p_high'] = [x[2] for x in mort_rates_c]
mort_c['p_scaled'] = mort_c['p_mean'] * death_adjt

ferguson = pd.read_excel('./data/ferguson.xlsx',sheet_name='Sheet1')

mort = mort.merge(ferguson[['age_group','ifr_from_hospital']],right_on='age_group',left_on='10yr_ageband')
mort.rename(columns={'ifr_from_hospital':'Ferguson_p'},inplace=True)

from scipy.optimize import curve_fit
def gompertz(x,a,b,c):
    return (a*np.exp(-b*np.exp(-c*x)))

params, pcov = curve_fit(gompertz, xdata=mort['Mid_age'], ydata=mort['p_scaled'], p0=[0.6,10,0.05])
mort['Gompertz'] = gompertz(mort['Mid_age'],params[0],params[1],params[2])
params, pcov = curve_fit(gompertz, xdata=mort_h['Mid_age'], ydata=mort_h['p_scaled'], p0=[0.6,20,0.05])
mort_h['Gompertz'] = gompertz(mort_h['Mid_age'],params[0],params[1],params[2])
params, pcov = curve_fit(gompertz, xdata=mort_c['Mid_age'], ydata=mort_c['p_scaled'], p0=[0.7,20,0.1])
mort_c['Gompertz'] = gompertz(mort_c['Mid_age'],params[0],params[1],params[2])


fig,ax = plt.subplots(1,1)

ax.plot(mort['p_mean'],c='darkblue',label='WC estimate')
ax.set_xlabel('Ageband')
ax.set_xticks(mort.index)
ax.set_xticklabels(mort['10yr_ageband'])
ax.plot(mort['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
ax.plot(mort['p_high'],c='darkblue',linestyle='dashed')
ax.plot(mort['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
ax.plot(mort['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
ax.plot(mort['Ferguson_p'],c='red',linestyle='dashdot',label='Ferguson')
ax.fill_between(mort.index,mort['p_low'],mort['p_high'],alpha=0.5,color='darkgray')

ax.set_title('Comparison of Western Cape Covid-19 mortality rate estimates by age with Ferguson et al. (2020)')
ax.legend()

fig.savefig('./data/Mortality_rates_20200616.png')

fig,ax = plt.subplots(2,1)

ax[0].plot(mort_h['p_mean'],c='darkblue',label='WC estimate')
ax[0].set_xlabel('Ageband')
ax[0].set_xticks(mort_h.index)
ax[0].set_xticklabels(mort_h['10yr_ageband'])
ax[0].plot(mort_h['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
ax[0].plot(mort_h['p_high'],c='darkblue',linestyle='dashed')
ax[0].fill_between(mort_h.index,mort_h['p_low'],mort_h['p_high'],alpha=0.5,color='darkgray')
ax[0].plot(mort_h['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
ax[0].plot(mort_h['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
ax[0].set_title('Mortality rates out of hospital')
ax[0].legend()

ax[1].plot(mort_c['p_mean'],c='darkblue',label='WC estimate')
ax[1].set_xlabel('Ageband')
ax[1].set_xticks(mort_c.index)
ax[1].set_xticklabels(mort_c['10yr_ageband'])
ax[1].plot(mort_c['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
ax[1].plot(mort_c['p_high'],c='darkblue',linestyle='dashed')
ax[1].fill_between(mort_c.index,mort_c['p_low'],mort_c['p_high'],alpha=0.5,color='darkgray')
ax[1].plot(mort_c['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
ax[1].plot(mort_c['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
ax[1].set_title('Mortality rates out of ICU')
ax[1].legend()

fig.tight_layout()

fig.savefig('./data/Mortality_rates_hosp_ICU_20200616.png')

mort.to_csv('./data/mort.csv')
mort_h.to_csv('./data/mort_h.csv')
mort_c.to_csv('./data/mort_c.csv')


