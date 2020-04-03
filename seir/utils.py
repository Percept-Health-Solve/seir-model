import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# make figures
sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (13, 8)
mpl.rcParams['figure.dpi'] = 150

def plot_solution(solution, t, group=None):
    s_t, e_t, i_t, r_t, d_t = solution
    if group is None:
        fig, axes = plt.subplots(2, 2, sharex=True)

        axes[0, 0].plot(t, np.sum(i_t, axis=(1, 2)) / 1000000, label='Total Infected')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 0], axis=-1) / 1000000, label='Infected Asymptomatic')
        axes[0, 0].plot(t, np.sum(i_t[:, :, 1], axis=-1) / 1000000, label='Infected Symptomatic')
        axes[0, 0].set_ylabel("Infections ('000 000)")
        axes[0, 0].legend()

        # axes[0, 1].plot(t, np.sum(i_t[:, :, 1], axis=-1)/1000000, label='Infected GP Seeking')
        axes[0, 1].plot(t, np.sum(i_t[:, :, 2], axis=-1) / 100000, label='Infected Hospitalised')
        axes[0, 1].plot(t, np.sum(i_t[:, :, 3], axis=-1) / 100000, label='Infected ICU')
        axes[0, 1].set_ylabel("Serious cases ('00 000)")
        axes[0, 1].legend()

        axes[1, 0].plot(t, np.sum(r_t, axis=(1, 2)) / 1000000, label='Recovered Total')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 0], axis=-1) / 1000000, label='Recovered Asymptomatic')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 1], axis=-1) / 1000000, label='Recovered Symptomatic')
        axes[1, 0].plot(t, np.sum(r_t[:, :, 2:], axis=(1, 2)) / 1000000, label='Recovered Serious')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel("Recoveries ('000 000)")
        axes[1, 0].legend()

        axes[1, 1].plot(t, np.sum(d_t, axis=(1, 2)) / 100000, label='Deceased Total')
        axes[1, 1].plot(t, np.sum(d_t[:, :, 1], axis=-1) / 100000, label='Deceased Symptomatic')
        axes[1, 1].plot(t, np.sum(d_t[:, :, 2:], axis=(1, 2)) / 100000, label='Deceased Serious')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel("Deaths ('00 000)")
        axes[1, 1].legend()
    else:
        fig, axes = plt.subplots(2, 2, sharex=True)

        axes[0, 0].plot(t, np.sum(i_t[:, group, :], axis=-1) / 1000000, label='Total Infected')
        axes[0, 0].plot(t, i_t[:, group, 0] / 1000000, label='Infected Asymptomatic')
        axes[0, 0].plot(t, i_t[:, group, 1] / 1000000, label='Infected Symptomatic')
        axes[0, 0].set_ylabel("Infections ('000 000)")
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