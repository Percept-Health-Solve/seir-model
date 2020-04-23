import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution_ds

# read calibration data
# actual_infections = pd.read_excel('../../Calibration.xlsx',sheet_name='Confirmed cases')
# actual_infections['Date'] = [pd.to_datetime(x).date() for x in actual_infections['Date']]
# actual_hospitalisations = pd.read_excel('../../Calibration.xlsx',sheet_name='Hospitalisations')
# actual_hospitalisations['Date'] = [pd.to_datetime(x).date() for x in actual_hospitalisations['Date']]
# actual_deaths = pd.read_excel('../../Calibration.xlsx',sheet_name='Deaths')
# actual_deaths['Date'] = [pd.to_datetime(x).date() for x in actual_deaths['Date']]

# get s0 from file:
df = pd.read_csv('data/Startpop_2density_0comorbidity.csv') #, index_col=0)
df['density'] = df['density'].map({'High': 'high', 'Low': 'low'})
df['label'] = df['age'].str.lower() + '_' + df['sex'].str.lower() + '_' + df['density'].str.lower()
df_dict = df[['label', 'Population']].to_dict()
s_0 = {df_dict['label'][i]: df_dict['Population'][i] for i in df_dict['label'].keys()}

# variable parameters for front-end
asymptomatic_prop = 0.7             # 0.2-0.8       Proportion of infections which are asymptomatic
asymp_rel_infectivity = 0.4         # 0.3 - 1       Relative infectivity of asymptomatic cases
asymp_prop_imported = 0.0           # 0 - 0.8       Proportion of imported cases which are asymptomatic
r0 = 3.5                            # 1.5 - 5.5     Initial R0
lockdown_ratio = 0.50               # 0.25 - 0.8    Lockdown R0 as a proportion of initial R0
imported_scale = 1.30               # 0.5 - 2       Scaling factor for imported case
lockdown_period = 35                # 35, 42, 49, 56, 63, 70  Length of lockdown period in days
social_distancing_ratio = 0.7       # 0.5-1         Post-lockdown R0 as a proportion of initial R0
period_asymp = 10                   # 8-12          Average days of infectiousness for asymptomatic cases
period_mild_infect = 2.3            # 2-4           Average days of pre-isolation infectiousness for mildly symptomatic cases
period_severe_infect = 2.3          # 2-4           Average days of pre-isolation infectiousness for acutely symptomatic cases
period_hosp_if_not_icu = 8          # 6-10          Average length of stay in hospital for cases that do not progress to ICU
period_hosp_if_icu = 8              # 6-10          Average pre-ICU length of stay in hospital for cases that progress to ICU
period_icu_if_recover = 10          # 8-12          Average ICU length of stay for cases that recover
period_icu_if_die = 5               # 3-7           Average ICU length of stay for cases that die
mort_loading = 1.0                  # 0.5 - 1.5     Mortality loading relative to Ferguson et al. parameters
prop_mild_detected = 0.6            # 0.2 - 0.8     Proportion of mildy symptomatic cases that are tested and detected


def get_results_from_parameters(s_0,
                                asymptomatic_prop,
                                asymp_rel_infectivity,
                                asymp_prop_imported,
                                r0,
                                lockdown_ratio,
                                imported_scale,
                                lockdown_period,
                                social_distancing_ratio,
                                period_asymp,
                                period_mild_infect,
                                period_severe_infect,
                                period_hosp_if_not_icu,
                                period_hosp_if_icu,
                                period_icu_if_recover,
                                period_icu_if_die,
                                mort_loading,
                                prop_mild_detected):

  # hard-coded parameters
  period_severe_isolate = 6 - period_severe_infect

  # Ferguson et al. parameterisation
  ferguson = {'0-9':[0.001,0.05,0.00002],
              '10-19':[0.003,0.05,0.00006],
              '20-29':[0.012,0.05,0.0003],
              '30-39':[0.032,0.05,0.0008],
              '40-49':[0.049,0.063,0.0015],
              '50-59':[0.102,0.122,0.006],
              '60-69':[0.166,0.274,0.022],
              '70-79':[0.243,0.432,0.051],
              '80+':[0.273,0.709,0.093]}

  # work out deaths as % of those entering ICU
  for key in ferguson:
    ferguson[key].append(ferguson[key][2]/ferguson[key][1]/ferguson[key][0])

  # infection and imported functions
  infectious_func = lambda t: 1 if t < 11 else (1-(1-social_distancing_ratio)/11*(t-11)) if 11<= t < 22 else lockdown_ratio if 22 <= t < (22+lockdown_period) else social_distancing_ratio
  c = 1
  s = 0.0         # proportion of imported cases below 60 that are severe (1-s are mild); assume 100% 60+ are severe
  a = 0.25
  l = asymp_prop_imported/(1-asymp_prop_imported)
  x = c * imported_scale

  imported_func = lambda t: {'0-9_male_high':    [0.0101 * x * l * np.exp(a*t), 0.0101 * x * (1-s) * np.exp(a*t), 0.0101 * x * s * np.exp(a*t), 0, 0, 0],
                            '10-19_male_high':   [0.0101 * x * l * np.exp(a*t), 0.0101 * x * (1-s) * np.exp(a*t), 0.0101 * x * s * np.exp(a*t), 0, 0, 0],
                            '20-29_male_high':   [0.0657 * x * l * np.exp(a*t), 0.0657 * x * (1-s) * np.exp(a*t), 0.0657 * x * s * np.exp(a*t), 0, 0, 0],
                            '30-39_male_high':   [0.1768 * x * l * np.exp(a*t), 0.1768 * x * (1-s) * np.exp(a*t), 0.1768 * x * s * np.exp(a*t), 0, 0, 0],
                            '40-49_male_high':   [0.0960 * x * l * np.exp(a*t), 0.0960 * x * (1-s) * np.exp(a*t), 0.0960 * x * s * np.exp(a*t), 0, 0, 0],
                            '50-59_male_high':   [0.1717 * x * l * np.exp(a*t), 0.1717 * x * (1-s) * np.exp(a*t), 0.1717 * x * s * np.exp(a*t), 0, 0, 0],
                            '60-69_male_high':   [0.0758 * x * l * np.exp(a*t), 0,                                0.0758 * x * np.exp(a*t),     0, 0, 0],
                            '70-79_male_high':   [0.0202 * x * l * np.exp(a*t), 0,                                0.0202 * x * np.exp(a*t),     0, 0, 0],
                            '80+_male_high':     [0.0051 * x * l * np.exp(a*t), 0,                                0.0051 * x * np.exp(a*t),     0, 0, 0],
                            '0-9_female_high':   [0.0000 * x * l * np.exp(a*t), 0.0000 * x * (1-s) * np.exp(a*t), 0.0000 * x * s * np.exp(a*t), 0, 0, 0],
                            '10-19_female_high': [0.0101 * x * l * np.exp(a*t), 0.0101 * x * (1-s) * np.exp(a*t), 0.0101 * x * s * np.exp(a*t), 0, 0, 0],
                            '20-29_female_high': [0.0606 * x * l * np.exp(a*t), 0.0606 * x * (1-s) * np.exp(a*t), 0.0606 * x * s * np.exp(a*t), 0, 0, 0],
                            '30-39_female_high': [0.1111 * x * l * np.exp(a*t), 0.1111 * x * (1-s) * np.exp(a*t), 0.1111 * x * s * np.exp(a*t), 0, 0, 0],
                            '40-49_female_high': [0.0556 * x * l * np.exp(a*t), 0.0556 * x * (1-s) * np.exp(a*t), 0.0556 * x * s * np.exp(a*t), 0, 0, 0],
                            '50-59_female_high': [0.0657 * x * l * np.exp(a*t), 0.0657 * x * (1-s) * np.exp(a*t), 0.0657 * x * s * np.exp(a*t), 0, 0, 0],
                            '60-69_female_high': [0.0152 * x * l * np.exp(a*t), 0,                                0.0152 * x * np.exp(a*t),     0, 0, 0],
                            '70-79_female_high': [0.0303 * x * l * np.exp(a*t), 0,                                0.0303 * x * np.exp(a*t),     0, 0, 0],
                            '80+_female_high':   [0.0000 * x * l * np.exp(a*t), 0,                                0.0000 * x * np.exp(a*t),     0, 0, 0]
                            } if t < 22 else 0

  # define model
  model = MultiPopWrapper(
      pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                      'sex': ['male', 'female'],
                      'density': ['high', 'low']
                      },
      inf_labels=['AS', 'M', 'S', 'SI', 'H', 'ICU'],
      alpha={'0-9': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['0-9'][0]), (1-asymptomatic_prop) * ferguson['0-9'][0], 0, 0, 0],
            '10-19': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['10-19'][0]), (1-asymptomatic_prop) * ferguson['10-19'][0], 0, 0, 0],
            '20-29': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['20-29'][0]), (1-asymptomatic_prop) * ferguson['20-29'][0], 0, 0, 0],
            '30-39': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['30-39'][0]), (1-asymptomatic_prop) * ferguson['30-39'][0], 0, 0, 0],
            '40-49': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['40-49'][0]), (1-asymptomatic_prop) * ferguson['40-49'][0], 0, 0, 0],
            '50-59': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['50-59'][0]), (1-asymptomatic_prop) * ferguson['50-59'][0], 0, 0, 0],
            '60-69': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['60-69'][0]), (1-asymptomatic_prop) * ferguson['60-69'][0], 0, 0, 0],
            '70-79': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['70-79'][0]), (1-asymptomatic_prop) * ferguson['70-79'][0], 0, 0, 0],
            '80+': [asymptomatic_prop, (1-asymptomatic_prop) * (1-ferguson['80+'][0]), (1-asymptomatic_prop) * ferguson['80+'][0], 0, 0, 0]},
      t_inc=5.1,
      q_se=[asymp_rel_infectivity, 1, 1, 0, 0, 0],
      q_ii=[
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 1 / period_severe_infect, 0, 0, 0],
          [0, 0, -1 / period_severe_infect, 1 / period_severe_isolate, 0, 0],   # changed hospital LOS to 8 days after 
          [0, 0, 0, -1 / period_severe_isolate, 1 / period_hosp_if_icu, 0],     # discussion with Barry 14/4/20
          [0, 0, 0, 0, -1 / period_hosp_if_icu, 0]
      ],
      q_ir=[1 / period_asymp, 1 / period_mild_infect, 0, 0, 1 / period_hosp_if_not_icu, 1 / period_icu_if_recover],
      q_id=[0, 0, 0, 0, 0, 1 / period_icu_if_die],
      rho_delta={'0-9': [0, 0, 1, 1, ferguson['0-9'][1], 0],
                '10-19': [0, 0, 1, 1, ferguson['10-19'][1], 0], 
                '20-29': [0, 0, 1, 1, ferguson['20-29'][1], 0],
                '30-39': [0, 0, 1, 1, ferguson['30-39'][1], 0],
                '40-49': [0, 0, 1, 1, ferguson['40-49'][1], 0],
                '50-59': [0, 0, 1, 1, ferguson['50-59'][1], 0],
                '60-69': [0, 0, 1, 1, ferguson['60-69'][1], 0],
                '70-79': [0, 0, 1, 1, ferguson['70-79'][1], 0],
                '80+': [0, 0, 1, 1, ferguson['80+'][1], 0]},
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
      extend_vars=True
  )

  init_vectors = {
      's_0': s_0,
      'i_0': {'30-39_male_high': [0, 0, 0, 0, 0, 0]}
  }

  periods_per_day = 5
  t = np.linspace(0, 300, 300 * periods_per_day + 1)
  solution = model.solve(init_vectors, t, to_csv=False, fp=None)
  model.q_se = model.q_se * r0 / model.r_0
  solution = model.solve(init_vectors, t, to_csv=True, fp='data/solution_ds.csv')

  # print(model.r_0)

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

  df_total = pd.DataFrame(np.concatenate([[s_total], [e_total], [a_total], [i_total], [sev_total], [h_total], [icu_total], [r_total], [d_total], [detected_total]]).T,
                          columns=['S', 'E', 'Asymptomatic', 'Mild', 'Severe Total', 'Hospitalised', 'ICU', 'R', 'Dead', 'Cumulative Detected'])
  df_total['Time'] = t
  df_total = df_total[[x==int(x) for x in df_total['Time']]]
  df_total['Day'] = [datetime.date(2020,3,5) + datetime.timedelta(days=t) for t in df_total['Time']]
  df_total.drop(columns='Time', inplace=True)
  cols = list(df_total.columns)
  cols = [cols[-1]] + cols[:-1]
  df_total = df_total[cols]
  df_total['Cumulative Infections'] = df_total['Asymptomatic'] + df_total['Mild'] + df_total['Severe Total'] + df_total['R'] + df_total['Dead']
  df_total['IFR'] = df_total['Dead'] / df_total['Cumulative Infections']
  df_total['CFR'] = df_total['Dead'] / df_total['Cumulative Detected']
  df_total['Total hospitalised'] = [a+b for a,b in zip(df_total['Hospitalised'],df_total['ICU'])]
  #df_total.to_csv('data/daily_output.csv',index=False)

  return(df_total)

  # plot output
  # fig, axes = plot_solution_ds(df_total,actual_infections,actual_hospitalisations,actual_deaths,45,90)
  # fig.savefig('data/output_'+descr+'.png')
