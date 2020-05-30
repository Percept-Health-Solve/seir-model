import pandas as pd 
import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 

sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (13, 8)
mpl.rcParams['figure.dpi'] = 100

df = pd.read_excel('./data/202005 WCDoH Covid19 admissions data v3.xlsx',sheet_name = 'WC Covid 19 Admissions')
print(f'Number of records in base file: {df.shape[0]:,}')
df = df[df['agegroup'] != 'Not recorded']
print(f'After dropping records with no age recorded: {df.shape[0]:,}')
df = df[df['admission_status'] != 'Inpatient']
print(f'After dropping current inpatients: {df.shape[0]:,}')
df['Count'] = 1

def tenyr_ageband(agegroup):
    lower = int((int(agegroup[:2])+2)/10)*10
    if lower >= 80:
        return('80+')
    else:
        return(str(lower) + ' - ' + str(lower+9))

df['10yr_ageband'] = df['agegroup'].apply(tenyr_ageband)

mort = pd.pivot_table(df,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
mort['Cases'] = mort['Died'] + mort['Discharged']

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

mort_rates = mort.apply(lambda row: conf_int(row['Died'], row['Cases']), axis=1)
mort['p_low'] = [x[0] for x in mort_rates]
mort['p_mean'] = [x[1] for x in mort_rates]
mort['p_high'] = [x[2] for x in mort_rates]

ferguson = pd.read_excel('./data/ferguson.xlsx',sheet_name='Sheet1')

mort = mort.merge(ferguson[['age_group','ifr_from_hospital']],right_on='age_group',left_on='10yr_ageband')
mort.rename(columns={'ifr_from_hospital':'Ferguson_p'},inplace=True)

fig,ax = plt.subplots(1,1)

ax.plot(mort['p_mean'],c='darkblue',label='WC estimate')
ax.set_xlabel('Ageband')
ax.set_xticks(mort.index)
ax.set_xticklabels(mort['10yr_ageband'])
ax.plot(mort['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
ax.plot(mort['p_high'],c='darkblue',linestyle='dashed')
ax.plot(mort['Ferguson_p'],c='red',linestyle='dashdot',label='Ferguson')
ax.fill_between(mort.index,mort['p_low'],mort['p_high'],alpha=0.5,color='darkgray')

ax.set_title('Comparison of Western Cape Covid-19 mortality rate estimates by age with Ferguson et al. (2020)')
ax.legend()

fig.savefig('./data/Mortality_rates.png')