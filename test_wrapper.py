import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution

infectious_func = lambda t: 1 if t < 22 else 0.25 if 22 <= t < 43 else 0.7 # scenario 1
# infectious_func = lambda t: 1 if t < 22 else 0.25 if 22 <= t < 64 else 1 # scenario 2
# imported_func = lambda t: [[0, 0.75 * 9 * np.exp(0.11*t), 0, 0], [0, 0.25 * 9 * np.exp(0.11*t), 5, 0]] if t < 16 else 0
c = 0.9
s = 0.25         # proportion of imported cases below 60 that are severe (1-s are mild); assume 100% 60+ are severe
a = 0.25
imported_func = lambda t: {'0-9_male_high': [0, 0.0101 * c * (1-s) * np.exp(a*t), 0.0101 * c * s * np.exp(a*t), 0, 0, 0],
                           '10-19_male_high': [0, 0.0101 * c * (1-s) * np.exp(a*t), 0.0101 * c * s * np.exp(a*t), 0, 0, 0],
                           '20-29_male_high': [0, 0.0657 * c * (1-s) * np.exp(a*t), 0.0657 * c * s * np.exp(a*t), 0, 0, 0],
                           '30-39_male_high': [0, 0.1768 * c * (1-s) * np.exp(a*t), 0.1768 * c * s * np.exp(a*t), 0, 0, 0],
                           '40-49_male_high': [0, 0.0960 * c * (1-s) * np.exp(a*t), 0.0960 * c * s * np.exp(a*t), 0, 0, 0],
                           '50-59_male_high': [0, 0.1717 * c * (1-s) * np.exp(a*t), 0.1717 * c * s * np.exp(a*t), 0, 0, 0],
                           '60-69_male_high': [0, 0, 0.0758 * c * (1-s) * np.exp(a*t), 0, 0, 0],
                           '70-79_male_high': [0, 0, 0.0202 * c * (1-s) * np.exp(a*t), 0, 0, 0],
                           '80+_male_high': [0, 0, 0.0051 * c * (1-s) * np.exp(a*t), 0, 0, 0],
                           '0-9_female_high': [0, 0 * c * (1-s) * np.exp(a * t), 0 * c * s * np.exp(a * t), 0, 0, 0],
                           '10-19_female_high': [0, 0.0101 * c * (1-s) * np.exp(a * t), 0.0101 * c * s * np.exp(a * t), 0, 0, 0],
                           '20-29_female_high': [0, 0.0606 * c * (1-s) * np.exp(a * t), 0.0606 * c * s * np.exp(a * t), 0, 0, 0],
                           '30-39_female_high': [0, 0.1111 * c * (1-s) * np.exp(a * t), 0.1111 * c * s * np.exp(a * t), 0, 0, 0],
                           '40-49_female_high': [0, 0.0556 * c * (1-s) * np.exp(a * t), 0.0556 * c * s * np.exp(a * t), 0, 0, 0],
                           '50-59_female_high': [0, 0.0657 * c * (1-s) * np.exp(a * t), 0.0657 * c * s * np.exp(a * t), 0, 0, 0],
                           '60-69_female_high': [0, 0, 0.0152 * c * (1-s) * np.exp(a * t), 0, 0, 0],
                           '70-79_female_high': [0, 0, 0.0303 * c * (1-s) * np.exp(a * t), 0, 0, 0],
                           '80+_female_high': [0, 0, 0 * c * (1-s) * np.exp(a * t), 0, 0, 0]
                           } if t < 22 else 0

model = MultiPopWrapper(
    pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                    'sex': ['male', 'female'],
                    'density': ['high', 'low']
                    },
    inf_labels=['AS', 'M', 'S', 'SI', 'H', 'ICU'],
    alpha={'0-9': [0.179, 0.821 * 0.999, 0.821 * 0.001, 0, 0, 0],
           '10-19': [0.179, 0.821 * 0.997, 0.821 * 0.003, 0, 0, 0],
           '20-29': [0.179, 0.821 * 0.988, 0.821 * 0.012, 0, 0, 0],
           '30-39': [0.179, 0.821 * 0.968, 0.821 * 0.032, 0, 0, 0],
           '40-49': [0.179, 0.821 * 0.951, 0.821 * 0.049, 0, 0, 0],
           '50-59': [0.179, 0.821 * 0.898, 0.821 * 0.102, 0, 0, 0],
           '60-69': [0.179, 0.821 * 0.834, 0.821 * 0.166, 0, 0, 0],
           '70-79': [0.179, 0.821 * 0.757, 0.821 * 0.243, 0, 0, 0],
           '80+': [0.179, 0.821 * 0.727, 0.821 * 0.273, 0, 0, 0]},
    t_inc=5.1,
    q_se=[0.45, 0.9, 0.9, 0, 0, 0],
    q_ii=[
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1 / 2.3, 0, 0, 0],
        [0, 0, -1 / 2.3, 1 / 2.7, 0, 0],
        [0, 0, 0, -1 / 2.7, 1 / 6, 0],
        [0, 0, 0, 0, -1 / 6, 0]
    ],
    q_ir=[1 / 10, 1 / 2.3, 0, 0, 1 / 8, 1 / 10],
    q_id=[0, 0, 0, 0, 0, 1 / 5],
    rho_delta={'0-9': [0, 0, 1, 1, 0.05, 0],
               '10-19': [0, 0, 1, 1, 0.03, 0],
               '20-29': [0, 0, 1, 1, 0.05, 0],
               '30-39': [0, 0, 1, 1, 0.05, 0],
               '40-49': [0, 0, 1, 1, 0.063, 0],
               '50-59': [0, 0, 1, 1, 0.122, 0],
               '60-69': [0, 0, 1, 1, 0.274, 0],
               '70-79': [0, 0, 1, 1, 0.432, 0],
               '80+': [0, 0, 1, 1, 0.709, 0]},
    rho_beta={'0-9': [0, 0, 0, 0, 0, 0.487],
              '10-19': [0, 0, 0, 0, 0, 0.487],
              '20-29': [0, 0, 0, 0, 0, 0.609],
              '30-39': [0, 0, 0, 0, 0, 0.609],
              '40-49': [0, 0, 0, 0, 0, 0.592],
              '50-59': [0, 0, 0, 0, 0, 0.587],
              '60-69': [0, 0, 0, 0, 0, 0.589],
              '70-79': [0, 0, 0, 0, 0, 0.592],
              '80+': [0, 0, 0, 0, 0, 0.585]},
    infectious_func=infectious_func,
    imported_func=imported_func,
    extend_vars=True
)

# get s0 from file:
df = pd.read_csv('data/Startpop_2density_0comorbidity.csv') #, index_col=0)
df['density'] = df['density'].map({'High': 'high', 'Low': 'low'})
df['label'] = df['age'].str.lower() + '_' + df['sex'].str.lower() + '_' + df['density'].str.lower()
df_dict = df[['label', 'Population']].to_dict()
s_0 = {df_dict['label'][i]: df_dict['Population'][i] for i in df_dict['label'].keys()}

init_vectors = {
    's_0': s_0,
    'i_0': {'30-39_male_high': [0, 0, 0, 0, 0, 0]}
}

periods_per_day = 5
t = np.linspace(0, 300, 300 * periods_per_day + 1)
solution = model.solve(init_vectors, t, to_csv=True, fp='data/solution.csv')

print(model.r_0)

s_t, e_t, i_t, r_t, d_t = solution

# plot all figures
fig, axes = plot_solution(solution, t, show_detected=True)

# all time plot
axes[1, 0].set_xticks((0, 50, 100, 150, 200, 250, 300))
axes[1, 0].set_xticklabels(('05-Mar', '24-Apr', '13-Jun', '02-Aug', '21-Sep', '10-Nov', '30-Dec'))

# 90 day plot
# for row in axes:
#     for ax in row:
#         ax.set_xlim((0, 90))
#         ax.set_ylim((0, 2000))
#         ax.axvline(x=22, color='k', ls='--', lw=0.8)
#         ax.axvline(x=64, color='k', ls='--', lw=0.8)
#
# axes[0, 1].set_ylim((0, 100))
# axes[1, 1].set_ylim((0, 50))
# axes[1, 0].set_xticklabels(('05-Mar', '15-Mar', '25-Mar', '04-Apr', '14-Apr', '24-Apr', '04-May', '14-May', '24-May', '03-Jun'))

plt.show()

s_total = np.sum(s_t, axis=-1)
e_total = np.sum(e_t, axis=-1)
a_total = np.sum(i_t[:, :, 0], axis=-1)
i_total = np.sum(i_t[:, :, 1], axis=-1)
sev_total = np.sum(i_t[:, :, 2:], axis=(1, 2))
h_total = np.sum(i_t[:, :, -2], axis=-1)
icu_total = np.sum(i_t[:, :, -1], axis=-1)
detected_total = 0.6 * np.sum(i_t[:, :, 1] + r_t[:, :, 1] + d_t[:, :, 1], axis=-1) \
                + np.sum(i_t[:, :, 2:] + r_t[:, :, 2:] + d_t[:, :, 2:], axis=(1, 2))
r_total = np.sum(r_t, axis=(1, 2))
d_total = np.sum(d_t, axis=(1, 2))

df_total = pd.DataFrame(np.concatenate([[s_total], [e_total], [a_total], [i_total], [sev_total], [h_total], [icu_total], [r_total], [d_total], [detected_total]]).T,
                        columns=['S', 'E', 'Asymptomatic', 'Mild', 'Severe Total', 'Hospitalised', 'ICU', 'R', 'Dead', 'Cumulative Detected'])
df_total['Time'] = t
df_total['Day'] = np.floor(df_total['Time'])
# df_total = df_total[df_total['Day'] <= 90]
df_total = df_total.groupby('Day').sum() / periods_per_day
df_total.drop(columns='Time', inplace=True)
df_total['Cumulative Infections'] = df_total['Asymptomatic'] + df_total['Mild'] + df_total['Severe Total'] + df_total['R'] + df_total['Dead']
df_total.to_csv('data/daily.csv')
