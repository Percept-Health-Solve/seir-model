import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime

from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution

sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (13, 8)
mpl.rcParams['figure.dpi'] = 100

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

# key parameters
param_dict = {'base':[1],
              'imported_case_scaling':[0.9,1.1],    # 1
              'pre_lockdown_post_disaster':[1,0.7], # lambda x:(1-0.3/11*(x-11))],
              'in_lockdown':[0.1,0.4],              # 0.25
              'post_lockdown':[0.5,1],              # 0.7
              'r0_base_case':[2.25,2.75],           # 2.5... solve for q_se
              'prop_asymptomatic':[0.25,0.4]}       # 0.179

for key in param_dict:

  for val in param_dict[key]:

    imp_case_scale = 1
    pre_lock = 1
    in_lock = 0.25
    post_lock = 0.7
    r0 = 2.5
    asymp = 0.179

    if key == 'imported_case_scaling':
      imp_case_scale = val
    elif key == 'pre_lockdown_post_disaster':
      pre_lock = val
    elif key == 'in_lockdown':
      in_lock = val
    elif key == 'post-lockdown':
      post_lock = val
    elif key == 'r0_base_case':
      r0 = val
    elif key == 'prop_asymptomatic':
      asymp = val
    
    infectious_func = lambda t: 1 if t < 11 else (1-(1-pre_lock)*(t-11)/11) if 11 <= t < 22 else in_lock if 22 <= t < 43 else post_lock # scenario 1  
    c = 0.9 * imp_case_scale
    s = 0.0         # proportion of imported cases below 60 that are severe (1-s are mild); assume 100% 60+ are severe
    a = 0.25
    imported_func = lambda t: {'0-9_male_high': [0, 0.0101 * c * (1-s) * np.exp(a*t), 0.0101 * c * s * np.exp(a*t), 0, 0, 0],
                              '10-19_male_high': [0, 0.0101 * c * (1-s) * np.exp(a*t), 0.0101 * c * s * np.exp(a*t), 0, 0, 0],
                              '20-29_male_high': [0, 0.0657 * c * (1-s) * np.exp(a*t), 0.0657 * c * s * np.exp(a*t), 0, 0, 0],
                              '30-39_male_high': [0, 0.1768 * c * (1-s) * np.exp(a*t), 0.1768 * c * s * np.exp(a*t), 0, 0, 0],
                              '40-49_male_high': [0, 0.0960 * c * (1-s) * np.exp(a*t), 0.0960 * c * s * np.exp(a*t), 0, 0, 0],
                              '50-59_male_high': [0, 0.1717 * c * (1-s) * np.exp(a*t), 0.1717 * c * s * np.exp(a*t), 0, 0, 0],
                              '60-69_male_high': [0, 0, 0.0758 * c * np.exp(a*t), 0, 0, 0],
                              '70-79_male_high': [0, 0, 0.0202 * c * np.exp(a*t), 0, 0, 0],
                              '80+_male_high': [0, 0, 0.0051 * c * np.exp(a*t), 0, 0, 0],
                              '0-9_female_high': [0, 0 * c * (1-s) * np.exp(a * t), 0 * c * s * np.exp(a * t), 0, 0, 0],
                              '10-19_female_high': [0, 0.0101 * c * (1-s) * np.exp(a * t), 0.0101 * c * s * np.exp(a * t), 0, 0, 0],
                              '20-29_female_high': [0, 0.0606 * c * (1-s) * np.exp(a * t), 0.0606 * c * s * np.exp(a * t), 0, 0, 0],
                              '30-39_female_high': [0, 0.1111 * c * (1-s) * np.exp(a * t), 0.1111 * c * s * np.exp(a * t), 0, 0, 0],
                              '40-49_female_high': [0, 0.0556 * c * (1-s) * np.exp(a * t), 0.0556 * c * s * np.exp(a * t), 0, 0, 0],
                              '50-59_female_high': [0, 0.0657 * c * (1-s) * np.exp(a * t), 0.0657 * c * s * np.exp(a * t), 0, 0, 0],
                              '60-69_female_high': [0, 0, 0.0152 * c * np.exp(a * t), 0, 0, 0],
                              '70-79_female_high': [0, 0, 0.0303 * c * np.exp(a * t), 0, 0, 0],
                              '80+_female_high': [0, 0, 0 * c * np.exp(a * t), 0, 0, 0]
                              } if t < 22 else 0

    model = MultiPopWrapper(
        pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                        'sex': ['male', 'female'],
                        'density': ['high', 'low']
                        },
        inf_labels=['AS', 'M', 'S', 'SI', 'H', 'ICU'],
        alpha={'0-9': [asymp, (1-asymp) * 0.999, (1-asymp) * 0.001, 0, 0, 0],
              '10-19': [asymp, (1-asymp) * 0.997, (1-asymp) * 0.003, 0, 0, 0],
              '20-29': [asymp, (1-asymp) * 0.988, (1-asymp) * 0.012, 0, 0, 0],
              '30-39': [asymp, (1-asymp) * 0.968, (1-asymp) * 0.032, 0, 0, 0],
              '40-49': [asymp, (1-asymp) * 0.951, (1-asymp) * 0.049, 0, 0, 0],
              '50-59': [asymp, (1-asymp) * 0.898, (1-asymp) * 0.102, 0, 0, 0],
              '60-69': [asymp, (1-asymp) * 0.834, (1-asymp) * 0.166, 0, 0, 0],
              '70-79': [asymp, (1-asymp) * 0.757, (1-asymp) * 0.243, 0, 0, 0],
              '80+': [asymp, (1-asymp) * 0.727, (1-asymp) * 0.273, 0, 0, 0]},
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

    periods_per_day = 5
    t = np.linspace(0, 300, 300 * periods_per_day + 1)
    solution = model.solve(init_vectors, t, to_csv=False, fp=None)
    model.q_se = model.q_se * r0 / model.r_0
    solution = model.solve(init_vectors, t, to_csv=False, fp=None)

    s_t, e_t, i_t, r_t, d_t = solution

    # plot all figures
    #fig, axes = plot_solution(solution, t, show_detected=True)

    # all time plot
    #axes[1, 0].set_xticks((0, 50, 100, 150, 200, 250, 300))
    #axes[1, 0].set_xticklabels(('05-Mar', '24-Apr', '13-Jun', '02-Aug', '21-Sep', '10-Nov', '30-Dec'))

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

    #plt.show()

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

    df_total = pd.DataFrame(np.concatenate([[t],[s_total], [e_total], [a_total], [i_total], [sev_total], [h_total], \
                                            [icu_total], [r_total], [d_total], [detected_total]]).T,
                            columns=['Time','S', 'E', 'Asymptomatic', 'Mild', 'Severe Total', 'Hospitalised', \
                                      'ICU', 'R', 'Dead', 'Cumulative Detected'])
    #df_total['Time'] = t
    df_total = df_total[[x==int(x) for x in df_total['Time']]]
    df_total['Day'] = [datetime.date(2020,3,5) + datetime.timedelta(days=t) for t in df_total['Time']]
    # df_total = df_total[df_total['Day'] <= 90]
    #df_total = df_total.groupby('Day').sum() / periods_per_day
    df_total.drop(columns='Time', inplace=True)
    cols = list(df_total.columns)
    cols = [cols[-1]] + cols[:-1]
    df_total = df_total[cols]
    df_total['Cumulative Infections'] = df_total['Asymptomatic'] + df_total['Mild'] + df_total['Severe Total'] + df_total['R'] + df_total['Dead']
    df_total['IFR'] = df_total['Dead'] / df_total['Cumulative Infections']
    df_total['CFR'] = df_total['Dead'] / df_total['Cumulative Detected']
    #df_total.to_csv('data/daily.csv')

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(
        df_total['Day'][:90],
        df_total['Cumulative Infections'][:90],
        label='Cumulative infections'
    )
    axes[0, 0].plot(
        df_total['Day'][:90],
        df_total['Cumulative Detected'][:90],
        label='Cumulative detected'
    )
    axes[0, 0].set_ylabel("Infections")
    axes[0, 0].legend()

    axes[0, 1].plot(
        df_total['Day'],
        df_total['Cumulative Infections'],
        label='Cumulative infections'
    )
    axes[0, 1].plot(
        df_total['Day'],
        df_total['Cumulative Detected'],
        label='Cumulative detected'
    )
    axes[0, 1].set_ylabel("Infections")
    axes[0, 1].legend()

    axes[1, 0].plot(
        df_total['Day'][:90],
        df_total['Hospitalised'][:90],
        label='Hospitalised (non-ICU)'
    )
    axes[1, 0].plot(
        df_total['Day'][:90],
        df_total['ICU'][:90],
        label='ICU'
    )
    axes[1, 0].set_ylabel("Hospitalisations")
    axes[1, 0].legend()

    axes[1, 1].plot(
        df_total['Day'],
        df_total['Hospitalised'],
        label='Hospitalised (non-ICU)'
    )
    axes[1, 1].plot(
        df_total['Day'],
        df_total['ICU'],
        label='ICU'
    )
    axes[1, 1].set_ylabel("Hospitalisations")
    axes[1, 1].legend()

    # all time plot
    axes[0, 0].set_xticks((datetime.date(2020,3,5),datetime.date(2020,3,20),datetime.date(2020,4,4),datetime.date(2020,4,19),
                            datetime.date(2020,5,4),datetime.date(2020,5,19),datetime.date(2020,6,3)))
    axes[0, 0].set_xticklabels(('05-Mar', '20-Mar', '04-Apr', '19-Apr', '04-May', '19-May', '03-Jun'))
    axes[0, 1].set_xticks((datetime.date(2020,3,5),datetime.date(2020,4,24),datetime.date(2020,6,13),datetime.date(2020,8,2),
                            datetime.date(2020,9,21),datetime.date(2020,11,10),datetime.date(2020,12,30)))
    axes[0, 1].set_xticklabels(('05-Mar', '24-Apr', '13-Jun', '02-Aug', '21-Sep', '10-Nov', '30-Dec'))
    axes[1, 0].set_xticks((datetime.date(2020,3,5),datetime.date(2020,3,20),datetime.date(2020,4,4),datetime.date(2020,4,19),
                            datetime.date(2020,5,4),datetime.date(2020,5,19),datetime.date(2020,6,3)))
    axes[1, 0].set_xticklabels(('05-Mar', '20-Mar', '04-Apr', '19_Apr', '04-May', '19-May', '03-Jun'))
    axes[1, 1].set_xticks((datetime.date(2020,3,5),datetime.date(2020,4,24),datetime.date(2020,6,13),datetime.date(2020,8,2),
                            datetime.date(2020,9,21),datetime.date(2020,11,10),datetime.date(2020,12,30)))
    axes[1, 1].set_xticklabels(('05-Mar', '24-Apr', '13-Jun', '02-Aug', '21-Sep', '10-Nov', '30-Dec'))
    
    if pre_lock == 1:
      pre_lock_str = 'Full'
    else:
      pre_lock_str = 'Tending to post-lockdown'

    main_title = 'R_0: ' + str(round(model.r_0,3)) + '; pre-lockdown R_0: ' + pre_lock_str \
                  + ';\n lockdown R_0 factor: ' + str(in_lock) \
                  + '; post-lockdown R_0 factor: ' + str(post_lock) \
                  + ';\n imported case scaling factor: ' + str(imp_case_scale) \
                  + '; asymptomatic proportion: ' + str(asymp)
    save_file = str(round(model.r_0,3)) + '_' + pre_lock_str + '_' + str(in_lock) \
                + '_' + str(post_lock) + '_' + str(imp_case_scale) + '_' + str(asymp) + '.png'
    fig.suptitle(main_title)
    plt.show()
    fig.savefig('data/' + save_file)

