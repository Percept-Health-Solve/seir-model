import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution

# read calibration data
actual_hospitalisations = pd.read_excel('data/calibration.xlsx', sheet_name='Hospitalisations')
actual_hospitalisations['Date'] = [pd.to_datetime(x, ).date() for x in actual_hospitalisations['Date']]

# TODO: should check if file is downloaded: if not, download, then use the downloaded file
actual_infections = pd.read_csv(
    'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_confirmed.csv')
actual_infections.rename(columns={'date': 'Date', 'total': 'Cum. Confirmed'}, inplace=True)
actual_infections.index = pd.to_datetime(actual_infections['Date'], dayfirst=True)
actual_infections = actual_infections.resample('D').mean().ffill().reset_index()
actual_infections['Date'] = [pd.to_datetime(x, dayfirst=True).date() for x in actual_infections['Date']]

# TODO: should check if file is downloaded: if not, download, then use the downloaded file
reported_deaths = pd.read_csv(
    'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_timeline_deaths.csv')
reported_deaths.rename(columns={'date': 'Date'}, inplace=True)
reported_deaths['Date'] = [pd.to_datetime(x, dayfirst=True).date() for x in reported_deaths['Date']]
actual_deaths = reported_deaths.groupby('Date').report_id.count().reset_index()
actual_deaths.rename(columns={'report_id': 'Daily deaths'}, inplace=True)
actual_deaths.index = pd.to_datetime(actual_deaths['Date'])
actual_deaths = actual_deaths.resample('D').mean().fillna(0).reset_index()
actual_deaths['Cum. Deaths'] = np.cumsum(actual_deaths['Daily deaths'])

# variable parameters for front-end
asymptomatic_prop = 0.75  # 0.2-0.8
asymp_rel_infectivity = 0.5  # 0.3 - 1
asymp_prop_imported = 0.0  # 0 - 0.8
r0 = 2.6  # 1.5 - 5.5
lockdown_ratio = 0.6  # 0.25 - 0.8
imported_scale = 2.5  # 0.5 - 2
lockdown_period = 35  # 35, 42, 49, 56, 63, 70
social_distancing_ratio = 0.75  # 0.5-1
period_asymp = 2.3  # 8-12
period_mild_infect = 2.3  # 2-4
period_severe_infect = 2.3  # 2-4
period_severe_isolate = 6 - period_severe_infect
period_hosp_if_not_icu = 10  # 6-10
period_hosp_if_icu = 8  # 6-10
period_icu_if_recover = 10  # 8-12
period_icu_if_die = 6  # 3-7
mort_loading = 1.0  # 0.5 - 1.5
prop_mild_detected = 0.3  # 0.2 - 0.8
hosp_to_icu = 0.2133  # 0.1 - 0.4 (0.21330242 = Ferguson)

descr = 'asymp_' + str(asymptomatic_prop) + '_R0_' + str(r0) + '_imported_scale_' + str(
    imported_scale) + '_lockdown_' + str(lockdown_ratio) + '_postlockdown_' + str(
    social_distancing_ratio) + '_ICU_' + str(hosp_to_icu) + '_mort_' + str(mort_loading) + '_asympinf_' + str(
    asymp_rel_infectivity)
full_descr = f'Baseline R0: {r0:.1f}, asymptomatic proportion: {asymptomatic_prop:.0%}, asymptomatic relative ' \
             f'infectiousness {asymp_rel_infectivity:.0%}, {prop_mild_detected:.0%} of mild cases detected \n '
full_descr += f'Imported scaling factor {imported_scale:.2f}, asymptomatic proportion imported {asymp_prop_imported:.0%}\n '
full_descr += f'Lockdown period: {lockdown_period:,.0f}, R0 relative to baseline {lockdown_ratio:.0%} in lockdown,' \
              f'{social_distancing_ratio:.0%} post-lockdown \n '
full_descr += f'Infectious days pre-isolation: {period_asymp} asymptomatic, {period_mild_infect} mild, {period_severe_infect} severe; severe isolation days pre-hospitalisation: {period_severe_isolate} \n'
full_descr += f'Hospital days: {period_hosp_if_not_icu} not critical, {period_hosp_if_icu} critical plus {period_icu_if_recover} in ICU if recover/{period_icu_if_die} if die \n'
full_descr += f'Proportion of hospitalised cases ending in ICU: {hosp_to_icu:.2%}, mortality loading {mort_loading:.0%}'

# get s0 from file:
df = pd.read_csv('data/Startpop_2density_0comorbidity.csv')  # , index_col=0)
df['density'] = df['density'].map({'High': 'high', 'Low': 'low'})
df['label'] = df['age'].str.lower() + '_' + df['sex'].str.lower() + '_' + df['density'].str.lower()
df_dict = df[['label', 'Population']].to_dict()
s_0 = {df_dict['label'][i]: df_dict['Population'][i] for i in df_dict['label'].keys()}

# Ferguson et al. parameterisation
ferguson = {'0-9': [0.001, 0.05, 0.00002],
            '10-19': [0.003, 0.05, 0.00006],
            '20-29': [0.012, 0.05, 0.0003],
            '30-39': [0.032, 0.05, 0.0008],
            '40-49': [0.049, 0.063, 0.0015],
            '50-59': [0.102, 0.122, 0.006],
            '60-69': [0.166, 0.274, 0.022],
            '70-79': [0.243, 0.432, 0.051],
            '80+': [0.273, 0.709, 0.093]}

# work out deaths as % of those entering ICU
for key in ferguson:
    # TODO: add this calc to the df, not to the lists.
    ferguson[key].append(ferguson[key][2] / ferguson[key][1] / ferguson[key][0])

# age profile - calculate ICU transition adjustment
age_profile = df.groupby('age').Population.sum().reset_index()
ferguson_df = pd.DataFrame(ferguson).T.reset_index()
ferguson_df.rename(columns={'index': 'age', 0: 'symp_to_hosp', 1: 'hosp_to_icu', 2: 'symp_to_dead', 3: 'icu_to_dead'},
                   inplace=True)
age_profile['Proportion'] = age_profile['Population'] / age_profile['Population'].sum()
age_profile = age_profile.merge(ferguson_df[['age', 'symp_to_hosp', 'hosp_to_icu']], on='age')
age_profile['hosp'] = age_profile['Proportion'] * age_profile['symp_to_hosp']
age_profile['prop_hosp'] = age_profile['hosp'] / age_profile['hosp'].sum()
age_profile['overall_hosp_to_icu'] = age_profile['prop_hosp'] * age_profile['hosp_to_icu']
overall_hosp_to_icu = age_profile['overall_hosp_to_icu'].sum()
icu_adjustment = hosp_to_icu / overall_hosp_to_icu  # ~1 when hosp_to_icu is == ferguson number

# hard-coded parameters

infectious_func = lambda t: 1 if t < 11 else (
            1 - (1 - social_distancing_ratio) / 11 * (t - 11)) if 11 <= t < 22 else lockdown_ratio if 22 <= t < (
            22 + lockdown_period) else social_distancing_ratio
c = 1
s = 0.06  # proportion of imported cases below 60 that are severe (1-s are mild)
# scale of ferguson ratio for 60+ - setting to inverse value from ferguson means we assume 100% of cases 60+ are severe
scale = {'60-69': 1,
         '70-79': 1/ferguson['70-79'][0],
         '80+': 1/ferguson['80+'][0]}
a = 0.25
l = asymp_prop_imported / (1 - asymp_prop_imported)
x = c * imported_scale

imported_func = lambda t: {'0-9_male_high': [0.0101 * x * l * np.exp(a * t), 0.0101 * x * (1 - s) * np.exp(a * t),
                                             0.0101 * x * s * np.exp(a * t), 0, 0, 0],
                           '10-19_male_high': [0.0101 * x * l * np.exp(a * t), 0.0101 * x * (1 - s) * np.exp(a * t),
                                               0.0101 * x * s * np.exp(a * t), 0, 0, 0],
                           '20-29_male_high': [0.0657 * x * l * np.exp(a * t), 0.0657 * x * (1 - s) * np.exp(a * t),
                                               0.0657 * x * s * np.exp(a * t), 0, 0, 0],
                           '30-39_male_high': [0.1768 * x * l * np.exp(a * t), 0.1768 * x * (1 - s) * np.exp(a * t),
                                               0.1768 * x * s * np.exp(a * t), 0, 0, 0],
                           '40-49_male_high': [0.0960 * x * l * np.exp(a * t), 0.0960 * x * (1 - s) * np.exp(a * t),
                                               0.0960 * x * s * np.exp(a * t), 0, 0, 0],
                           '50-59_male_high': [0.1717 * x * l * np.exp(a * t), 0.1717 * x * (1 - ferguson['50-59'][0]) * np.exp(a * t),
                                               0.1717 * x * ferguson['50-59'][0] * np.exp(a * t), 0, 0, 0],
                           '60-69_male_high': [0.0758 * x * l * np.exp(a * t), 0.0758 * x * (1 - scale['60-69'] * ferguson['60-69'][0]) * np.exp(a * t), 0.0758 * x * scale['60-69'] * ferguson['60-69'][0] * np.exp(a * t), 0, 0, 0],
                           '70-79_male_high': [0.0202 * x * l * np.exp(a * t), 0.0202 * x * (1 - scale['70-79'] * ferguson['70-79'][0]) * np.exp(a * t), 0.0202 * x * scale['70-79'] * ferguson['70-79'][0] * np.exp(a * t), 0, 0, 0],
                           '80+_male_high': [0.0051 * x * l * np.exp(a * t), 0.0051 * x * (1 - scale['80+'] * ferguson['80+'][0]) * np.exp(a * t), 0.0051 * x * scale['80+'] * ferguson['80+'][0] * np.exp(a * t), 0, 0, 0],
                           '0-9_female_high': [0.0000 * x * l * np.exp(a * t), 0.0000 * x * (1 - s) * np.exp(a * t),
                                               0.0000 * x * s * np.exp(a * t), 0, 0, 0],
                           '10-19_female_high': [0.0101 * x * l * np.exp(a * t), 0.0101 * x * (1 - s) * np.exp(a * t),
                                                 0.0101 * x * s * np.exp(a * t), 0, 0, 0],
                           '20-29_female_high': [0.0606 * x * l * np.exp(a * t), 0.0606 * x * (1 - s) * np.exp(a * t),
                                                 0.0606 * x * s * np.exp(a * t), 0, 0, 0],
                           '30-39_female_high': [0.1111 * x * l * np.exp(a * t), 0.1111 * x * (1 - s) * np.exp(a * t),
                                                 0.1111 * x * s * np.exp(a * t), 0, 0, 0],
                           '40-49_female_high': [0.0556 * x * l * np.exp(a * t), 0.0556 * x * (1 - s) * np.exp(a * t),
                                                 0.0556 * x * s * np.exp(a * t), 0, 0, 0],
                           '50-59_female_high': [0.0657 * x * l * np.exp(a * t), 0.0657 * x * (1 - s) * np.exp(a * t),
                                                 0.0657 * x * s * np.exp(a * t), 0, 0, 0],
                           '60-69_female_high': [0.0152 * x * l * np.exp(a * t), 0.0152 * x * (1 - scale['60-69'] * ferguson['60-69'][0]) * np.exp(a * t), 0.0152 * x * scale['60-69'] * ferguson['60-69'][0] * np.exp(a * t), 0, 0,
                                                 0],
                           '70-79_female_high': [0.0303 * x * l * np.exp(a * t), 0.0303 * x * (1 - scale['70-79'] * ferguson['70-79'][0]) * np.exp(a * t), 0.0303 * x * scale['70-79'] * ferguson['70-79'][0] * np.exp(a * t), 0, 0,
                                                 0],
                           '80+_female_high': [0.0000 * x * l * np.exp(a * t), 0.0000 * x * (1 - scale['80+'] * ferguson['80+'][0]) * np.exp(a * t), 0.0000 * x * scale['80+'] * ferguson['80+'][0] * np.exp(a * t), 0, 0, 0]
                           } if t < 22 else 0

init_vectors = {
    's_0': s_0,
    'i_0': {'30-39_male_high': [0, 0, 0, 0, 0, 0]}
}

model = MultiPopWrapper(
    pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                    'sex': ['male', 'female'],
                    'density': ['high', 'low']
                    },
    inf_labels=['AS', 'M', 'S', 'SI', 'H', 'ICU'],
    alpha={'0-9': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['0-9'][0]),
                   (1 - asymptomatic_prop) * ferguson['0-9'][0], 0, 0, 0],
           '10-19': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['10-19'][0]),
                     (1 - asymptomatic_prop) * ferguson['10-19'][0], 0, 0, 0],
           '20-29': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['20-29'][0]),
                     (1 - asymptomatic_prop) * ferguson['20-29'][0], 0, 0, 0],
           '30-39': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['30-39'][0]),
                     (1 - asymptomatic_prop) * ferguson['30-39'][0], 0, 0, 0],
           '40-49': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['40-49'][0]),
                     (1 - asymptomatic_prop) * ferguson['40-49'][0], 0, 0, 0],
           '50-59': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['50-59'][0]),
                     (1 - asymptomatic_prop) * ferguson['50-59'][0], 0, 0, 0],
           '60-69': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['60-69'][0]),
                     (1 - asymptomatic_prop) * ferguson['60-69'][0], 0, 0, 0],
           '70-79': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['70-79'][0]),
                     (1 - asymptomatic_prop) * ferguson['70-79'][0], 0, 0, 0],
           '80+': [asymptomatic_prop, (1 - asymptomatic_prop) * (1 - ferguson['80+'][0]),
                   (1 - asymptomatic_prop) * ferguson['80+'][0], 0, 0, 0]},
    t_inc=5.1,
    q_se=[asymp_rel_infectivity, 1, 1, 0, 0, 0],
    q_ii=[
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1 / period_severe_infect, 0, 0, 0],
        [0, 0, -1 / period_severe_infect, 1 / period_severe_isolate, 0, 0],
        [0, 0, 0, -1 / period_severe_isolate, 1 / period_hosp_if_icu, 0],
        [0, 0, 0, 0, -1 / period_hosp_if_icu, 0]
    ],
    q_ir=[1 / period_asymp, 1 / period_mild_infect, 0, 0, 1 / period_hosp_if_not_icu, 1 / period_icu_if_recover],
    q_id=[0, 0, 0, 0, 0, 1 / period_icu_if_die],
    rho_delta={'0-9': [0, 0, 1, 1, ferguson['0-9'][1] * icu_adjustment, 0],
               '10-19': [0, 0, 1, 1, ferguson['10-19'][1] * icu_adjustment, 0],
               '20-29': [0, 0, 1, 1, ferguson['20-29'][1] * icu_adjustment, 0],
               '30-39': [0, 0, 1, 1, ferguson['30-39'][1] * icu_adjustment, 0],
               '40-49': [0, 0, 1, 1, ferguson['40-49'][1] * icu_adjustment, 0],
               '50-59': [0, 0, 1, 1, ferguson['50-59'][1] * icu_adjustment, 0],
               '60-69': [0, 0, 1, 1, ferguson['60-69'][1] * icu_adjustment, 0],
               '70-79': [0, 0, 1, 1, ferguson['70-79'][1] * icu_adjustment, 0],
               '80+': [0, 0, 1, 1, ferguson['80+'][1] * icu_adjustment, 0]},
    rho_beta={'0-9': [0, 0, 0, 0, 0, ferguson['0-9'][3] * mort_loading],
              '10-19': [0, 0, 0, 0, 0, ferguson['10-19'][3] * mort_loading],
              '20-29': [0, 0, 0, 0, 0, ferguson['20-29'][3] * mort_loading],
              '30-39': [0, 0, 0, 0, 0, ferguson['30-39'][3] * mort_loading],
              '40-49': [0, 0, 0, 0, 0, ferguson['40-49'][3] * mort_loading],
              '50-59': [0, 0, 0, 0, 0, ferguson['50-59'][3] * mort_loading],
              '60-69': [0, 0, 0, 0, 0, ferguson['60-69'][3] * mort_loading],
              '70-79': [0, 0, 0, 0, 0, ferguson['70-79'][3] * mort_loading],
              '80+': [0, 0, 0, 0, 0, ferguson['80+'][3] * mort_loading]},
    infectious_func=infectious_func,
    imported_func=imported_func,
    init_vectors=init_vectors,
    extend_vars=True
)

periods_per_day = 5
t = np.linspace(0, 300, 300 * periods_per_day + 1)
model.q_se = model.q_se * r0 / model.r_0
print(model.q_se)
solution = model.solve(t, to_csv=True, fp='data/solution.csv')

s_t, e_t, i_t, r_t, d_t = solution
s_total = np.sum(s_t, axis=-1)
e_total = np.sum(e_t, axis=-1)
a_total = np.sum(i_t[:, :, 0], axis=-1)
i_total = np.sum(i_t[:, :, 1], axis=-1)
sev_total = np.sum(i_t[:, :, 2:], axis=(1, 2))
h_total = np.sum(i_t[:, :, -2], axis=-1)
icu_total = np.sum(i_t[:, :, -1], axis=-1)
detected_total = prop_mild_detected * np.sum(i_t[:, :, 1] + r_t[:, :, 1] + d_t[:, :, 1], axis=-1) \
                 + np.sum(i_t[:, :, 2:] + r_t[:, :, 2:] + d_t[:, :, 2:], axis=(1, 2))
r_total = np.sum(r_t, axis=(1, 2))
d_total = np.sum(d_t, axis=(1, 2))

df_total = pd.DataFrame(np.concatenate(
    [[s_total], [e_total], [a_total], [i_total], [sev_total], [h_total], [icu_total], [r_total], [d_total],
     [detected_total]]).T,
                        columns=['S', 'E', 'Asymptomatic', 'Mild', 'Severe Total', 'Hospitalised', 'ICU', 'R', 'Dead',
                                 'Cumulative Detected'])
df_total['Time'] = t
df_total = df_total[[x == int(x) for x in df_total['Time']]]
df_total['Day'] = [datetime.date(2020, 3, 5) + datetime.timedelta(days=t) for t in df_total['Time']]
df_total.drop(columns='Time', inplace=True)
cols = list(df_total.columns)
cols = [cols[-1]] + cols[:-1]
df_total = df_total[cols]
df_total['Cumulative Infections'] = df_total['Asymptomatic'] + df_total['Mild'] + df_total['Severe Total'] + df_total[
    'R'] + df_total['Dead']
df_total['IFR'] = df_total['Dead'] / df_total['Cumulative Infections']
df_total['CFR'] = df_total['Dead'] / df_total['Cumulative Detected']
df_total['Total hospitalised'] = [a + b for a, b in zip(df_total['Hospitalised'], df_total['ICU'])]
df_total['Active infections'] = df_total['Asymptomatic'] + df_total['Mild'] + df_total['Severe Total']
df_total.to_csv('data/daily_output_' + descr + '.csv', index=False)

# plot output
fig, axes = plot_solution(df_total, full_descr, actual_infections, actual_hospitalisations, actual_deaths, 45, 90)
fig.savefig('data/output_' + descr + '.png')

plt.stackplot(
    np.asarray(df_total['Day']),
    np.asarray(df_total['Active infections']), np.asarray(df_total['S']), np.asarray(df_total['E']),
    np.asarray(df_total['R']), np.asarray(df_total['Dead']),
    labels=['I', 'S', 'E', 'R', 'D'],
    colors=['red', 'lightgray', 'blue', 'green', 'black']
)
