import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import datetime

# make figures
sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (13, 8)
mpl.rcParams['figure.dpi'] = 100


def plot_solution(solution, t, group=None, show_cumulative=False, show_detected=True):
    s_t, e_t, i_t, r_t, d_t = solution
    if group is None:
        fig, axes = plt.subplots(2, 2, sharex=True)

        if show_cumulative:
            axes[0, 0].plot(
                t,
                np.sum(i_t[:, :, 1:] + r_t[:, :, 1:] + d_t[:, :, 1:], axis=(1, 2)),
                label='Cumulative Symptomatic Infections'
            )
        if show_detected:
            axes[0, 0].plot(
                t,
                0.6 * np.sum(i_t[:, :, 1] + r_t[:, :, 1] + d_t[:, :, 1], axis=-1)
                + np.sum(i_t[:, :, 2:] + r_t[:, :, 2:] + d_t[:, :, 2:], axis=(1, 2)),
                label='Detected Symptomatic Infections'
            )
        axes[0, 0].plot(t, np.sum(i_t, axis=(1, 2)), label='Total Infected')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 0], axis=-1), label='Infectious Asymptomatic')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 1], axis=-1), label='Infectious Mild')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 2], axis=-1), label='Infectious Severe')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 3:], axis=(1, 2)), label='Isolated Severe (I/H/ICU)')
        axes[0, 0].set_ylabel("Infections")
        axes[0, 0].legend()

        # axes[0, 1].plot(t, np.sum(i_t[:, :, 1], axis=-1)/1000000, label='Infected GP Seeking')
        axes[0, 1].plot(t, np.sum(i_t[:, :, -2], axis=-1), label='Infected Hospitalised')
        axes[0, 1].plot(t, np.sum(i_t[:, :, -1], axis=-1), label='Infected ICU')
        axes[0, 1].set_ylabel("Serious cases")
        axes[0, 1].legend()

        axes[1, 0].plot(t, np.sum(r_t, axis=(1, 2)), label='Isolated Total')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 0], axis=-1), label='Isolated Asymptomatic')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 1], axis=-1), label='Isolated Mild')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 2:], axis=(1, 2)), label='Isolated Severe')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel("Recoveries")
        axes[1, 0].legend()

        axes[1, 1].plot(t, np.sum(d_t, axis=(1, 2)), label='Deceased Total')
        axes[1, 1].plot(t, np.sum(d_t[:, :, 1], axis=-1), label='Deceased Mild')
        axes[1, 1].plot(t, np.sum(d_t[:, :, 2:], axis=(1, 2)), label='Deceased Severe')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel("Deaths")
        axes[1, 1].legend()
    else:
        fig, axes = plt.subplots(2, 2, sharex=True)

        axes[0, 0].plot(t, np.sum(i_t[:, group, :], axis=-1), label='Total Infected')
        axes[0, 0].plot(t, i_t[:, group, 0], label='Infected Asymptomatic')
        axes[0, 0].plot(t, i_t[:, group, 1], label='Infected Symptomatic')
        axes[0, 0].set_ylabel("Infections")
        axes[0, 0].legend()

        # axes[0, 1].plot(t, np.sum(i_t[:, :, 1], axis=-1)/1000000, label='Infected GP Seeking')
        axes[0, 1].plot(t, i_t[:, group, 2] / 100000, label='Infected Hospitalised')
        axes[0, 1].plot(t, i_t[:, group, 3] / 100000, label='Infected ICU')
        axes[0, 1].set_ylabel("Serious cases ('00 000)")
        axes[0, 1].legend()

        axes[1, 0].plot(t, np.sum(r_t[:, group, :], axis=-1) / 1000000, label='Recovered Total')
        axes[1, 0].plot(t, r_t[:, group, 0] / 1000000, label='Recovered Asymptomatic')
        axes[1, 0].plot(t, r_t[:, group, 1] / 1000000, label='Recovered Symptomatic')
        axes[1, 0].plot(t, np.sum(r_t[:, group, 2:], axis=-1) / 1000000, label='Recovered Serious')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel("Recoveries ('000 000)")
        axes[1, 0].legend()

        axes[1, 1].plot(t, np.sum(d_t[:, group, :], axis=-1) / 100000, label='Deceased Total')
        axes[1, 1].plot(t, d_t[:, group, 1] / 100000, label='Deceased Symptomatic')
        axes[1, 1].plot(t, np.sum(d_t[:, group, 2:], axis=-1) / 100000, label='Deceased Serious')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel("Deaths ('00 000)")
        axes[1, 1].legend()

    return fig, axes


def set_tick_points(t,divisions,start_date=str):
  ticks = [pd.to_datetime(start_date).date() + datetime.timedelta(days=t) for t in [round(t/divisions*x) for x in range(divisions+1)]]
  tick_labels = [x.strftime('%d-%b') for x in ticks]
  return(ticks,tick_labels)


def plot_solution_ds(df_soln,actual_infections,actual_hospitalisations,actual_deaths,t1,t2):

  sns.set(style='whitegrid')
  mpl.rcParams['figure.figsize'] = (18,20)
  mpl.rcParams['figure.dpi'] = 100

  fig, axes = plt.subplots(5, 3)

  axes[0, 0].plot(
    df_soln['Day'][:t1+1],
    df_soln['Cumulative Infections'][:t1+1],
    label='Cumulative infections'
  )
  axes[0, 1].plot(
      df_soln['Day'][:t2+1],
      df_soln['Cumulative Infections'][:t2+1],
      label='Cumulative infections'
  )
  axes[0, 2].plot(
      df_soln['Day'],
      df_soln['Cumulative Infections'],
      label='Cumulative infections'
  )

  axes[1, 0].plot(
      df_soln['Day'][:t1+1],
      df_soln['Cumulative Detected'][:t1+1],
      label='Cumulative detected'
  )
  axes[1, 1].plot(
      df_soln['Day'][:t2+1],
      df_soln['Cumulative Detected'][:t2+1],
      label='Cumulative detected'
  )
  axes[1,2].plot(
      df_soln['Day'],
      df_soln['Cumulative Detected'],
      label='Cumulative detected'
  )

  axes[2, 0].plot(
      df_soln['Day'][:t1+1],
      df_soln['Total hospitalised'][:t1+1],
      label='Total hospitalised'
  )
  axes[2, 1].plot(
      df_soln['Day'][:t2+1],
      df_soln['Total hospitalised'][:t2+1],
      label='Total hospitalised'
  )
  axes[2, 2].plot(
      df_soln['Day'],
      df_soln['Total hospitalised'],
      label='Total hospitalised'
  )

  axes[3, 0].plot(
      df_soln['Day'][:t1+1],
      df_soln['ICU'][:t1+1],
      label='ICU'
  )
  axes[3, 1].plot(
      df_soln['Day'][:t2+1],
      df_soln['ICU'][:t2+1],
      label='ICU'
  )
  axes[3, 2].plot(
      df_soln['Day'],
      df_soln['ICU'],
      label='ICU'
  )

  axes[4, 0].plot(
      df_soln['Day'][:t1+1],
      df_soln['Dead'][:t1+1],
      label='Cum. deaths'
  )
  axes[4, 1].plot(
      df_soln['Day'][:t2+1],
      df_soln['Dead'][:t2+1],
      label='Cum. deaths'
  )
  axes[4, 2].plot(
      df_soln['Day'],
      df_soln['Dead'],
      label='Cum. deaths'
  )

  for x in [0,1]:
    axes[1, x].plot(
        actual_infections['Date'],
        actual_infections['Cum. Confirmed'],
        label='Reported cases'
    )
    axes[2, x].plot(
        actual_hospitalisations['Date'],
        actual_hospitalisations['Private hospital'],
        label='Reported private'
    )
    axes[2, x].plot(
        actual_hospitalisations['Date'],
        actual_hospitalisations['Est. total hospital'],
        label='Estimated total'
    )
    axes[3, x].plot(
        actual_hospitalisations['Date'],
        actual_hospitalisations['Private ICU'],
        label='Reported ICU private'
    )
    axes[3, x].plot(
        actual_hospitalisations['Date'],
        actual_hospitalisations['Est. total ICU'],
        label='Estimated ICU total'
    )
    axes[4, x].plot(
        actual_deaths['Date'],
        actual_deaths['Cum. Deaths'],
        label='Reported deaths'
    )

  for i in range(3):

    axes[0, i].set_ylabel("Infections")
    axes[0, i].legend()
    axes[1, i].set_ylabel("Detected infections")
    axes[1, i].legend()
    axes[2, i].set_ylabel("Total hospitalisations")
    axes[2, i].legend()
    axes[3, i].set_ylabel("ICU")
    axes[3, i].legend()
    axes[4, i].set_ylabel("Cumulative deaths")
    axes[4, i].legend()

  t1_range = set_tick_points(t1,6,'2020-03-05')
  t2_range = set_tick_points(t2,6,'2020-03-05')
  t3_range = set_tick_points(df_soln.shape[0]-1,6,'2020-03-05')

  for i in range(5):
    axes[i, 0].set_xticks(t1_range[0])
    axes[i, 0].set_xticklabels(t1_range[1])
    axes[i, 1].set_xticks(t2_range[0])
    axes[i, 1].set_xticklabels(t2_range[1])
    axes[i, 2].set_xticks(t3_range[0])
    axes[i, 2].set_xticklabels(t3_range[1])

  return fig, axes