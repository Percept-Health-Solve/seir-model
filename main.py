import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')

# R0 = np.array([3, 3, 0, 0])
T_inc = 5
alpha_e = np.array([
    [0.6, 0.4, 0.0, 0.0],
    [0.4, 0.3, 0.3, 0.0]
])
delta = np.array([
    [0.02, 0.20, 0.04, 0],
    [0.02, 0.20, 0.04, 0],
])
beta = np.array([
    [0.00, 0.002, 0.002, 0.002],
    [0.00, 0.002, 0.002, 0.002]
])

Q_II = np.array([
    [1 / 10, 0, 0, 0],
    [0, 1 / 5, 0, 0],
    [-1 / 10, -1 / 5, 1 / 4, -1E-20],
    [0, 0, -1 / 4, 1E-20]
])
Q_IR = np.array([1 / 11, 1 / 2, 1 / 7, 1 / 14])
Q_ID = np.array([1 / 11, 1 / 2, 1 / 5, 1 / 5])

D_t = np.diag(Q_II) * delta + Q_IR * (1 - beta) * (1 - delta) + Q_ID * beta * (1 - delta)
# Q_S = R0 / D_t
Q_S = np.array([0.1, 0.434, 0, 0])  # interpretation is number of new infections per person per day

nb_groups = 2
S_0 = np.array([[25000000], [5000000]])
E_0 = np.zeros((nb_groups, 1))
I_0 = np.zeros((nb_groups, 4))
I_0[0, 1] = 100
R_0 = np.zeros((nb_groups, 4))
D_0 = np.zeros((nb_groups, 4))

y0 = np.concatenate([
    S_0.reshape(-1),
    E_0.reshape(-1),
    I_0.reshape(-1),
    R_0.reshape(-1),
    D_0.reshape(-1)
])

# y0 = np.array([20000000, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# y0 = y0 / np.sum(y0)
N = np.sum(y0)

print("D_t", D_t)
print("Q_S", Q_S)
print("R0", Q_S / np.average(D_t, axis=0, weights=[25000000, 5000000]) * np.average(alpha_e, axis=0, weights=[25000000, 5000000]), np.sum(Q_S / np.average(D_t, axis=0, weights=[25000000, 5000000]) * np.average(alpha_e, axis=0, weights=[25000000, 5000000]), axis=0), np.mean(np.sum(Q_S / D_t * alpha_e, axis=1)))
print("Eff R0", Q_S / D_t)
print(N)

SA_data = np.array([
    [0, 8],
    [1, 14],
    [2, 21],
    [3, 44],
    [4, 61],
    [5, 75]
])


def ode(y, t):
    S = y[0:nb_groups].reshape(nb_groups, -1)
    E = y[nb_groups:nb_groups * 2].reshape(nb_groups, -1)
    I = y[nb_groups * 2:nb_groups * 2 + nb_groups * 4].reshape(nb_groups, -1)
    R = y[nb_groups * 2 + nb_groups * 4:nb_groups * 2 + nb_groups * 4 * 2].reshape(nb_groups, -1)
    D = y[nb_groups * 2 + nb_groups * 4 * 2:].reshape(nb_groups, -1)

    dSdt = - 1 / N * Q_S.dot(np.sum(I, axis=0)) * S
    dEdt = 1 / N * Q_S.dot(np.sum(I, axis=0)) * S - E / T_inc
    dIdt = alpha_e * E / T_inc - (delta * I).dot(Q_II.T) - Q_IR * (1 - beta) * (1 - delta) * I - Q_ID * beta * (
                1 - delta) * I
    dRdt = Q_IR * (1 - beta) * (1 - delta) * I
    dDdt = Q_ID * beta * (1 - delta) * I

    out = np.concatenate([
        dSdt.reshape(-1),
        dEdt.reshape(-1),
        dIdt.reshape(-1),
        dRdt.reshape(-1),
        dDdt.reshape(-1)
    ])
    return out


t = np.linspace(0, 300, 10000)

sol = odeint(ode, y0, t)
S = sol[:, 0:nb_groups].reshape(10000, nb_groups)
E = sol[:, nb_groups:nb_groups * 2].reshape(10000, nb_groups)
I = sol[:, nb_groups * 2:nb_groups * 2 + nb_groups * 4].reshape(10000, nb_groups, -1)
R = sol[:, nb_groups * 2 + nb_groups * 4:nb_groups * 2 + nb_groups * 4 * 2].reshape(10000, nb_groups, -1)
D = sol[:, nb_groups * 2 + nb_groups * 4 * 2:].reshape(10000, nb_groups, -1)

df = pd.DataFrame(sol, columns=['S_0', 'S_1',
                                'E_0', 'E_1',
                                'I_0_AS', 'I_0_GP', 'I_0_H', 'I_0_ICU',
                                'I_1_AS', 'I_1_GP', 'I_1_H', 'I_1_ICU',
                                'R_0_AS', 'R_0_GP', 'R_0_H', 'R_0_ICU',
                                'R_1_AS', 'R_1_GP', 'R_1_H', 'R_1_ICU',
                                'D_0_AS', 'D_0_GP', 'D_0_H', 'D_0_ICU',
                                'D_1_AS', 'D_1_GP', 'D_1_H', 'D_1_ICU'])
df.to_csv('sol_1.csv')

plt.plot(t, np.sum(I, axis=(1, 2)), label='I')
plt.plot(t, np.sum(E, axis=1), label='E')
plt.plot(t, np.sum(I, axis=(1, 2)) + np.sum(E, axis=1), label='I+E')
plt.legend()
plt.show()

# plt.plot(t, I[:, 0, 0], label='I_AS')
# plt.plot(t, I[:, 0, 1], label='I_GP')
# plt.plot(t, I[:, 0, 2], label='I_H')
# plt.plot(t, I[:, 0, 3], label='I_ICU')
# plt.scatter(SA_data[:, 0], SA_data[:, 1], label='SA Data')
# plt.xlim(0, 20)
# plt.ylim(0, 2000)
# plt.legend()
# plt.show()
#
# # plt.plot(t, sol[:, 0], label='S')
# plt.plot(t, E[:, 0], label='E')
# plt.plot(t, I[:, 0, 0], label='I_AS')
# plt.plot(t, I[:, 0, 1], label='I_GP')
# plt.plot(t, I[:, 0, 2], label='I_H')
# plt.plot(t, I[:, 0, 3], label='I_ICU')
# plt.plot(t, np.sum(I[:, 0], axis=1), label='I')
# plt.plot(t, R[:, 0, 0], label='R_AS')
# plt.plot(t, R[:, 0, 1], label='R_GP')
# plt.plot(t, R[:, 0, 2], label='R_H')
# plt.plot(t, R[:, 0, 3], label='R_ICU')
# plt.plot(t, D[:, 0, 0], label='D_AS')
# plt.plot(t, D[:, 0, 1], label='D_GP')
# plt.plot(t, D[:, 0, 2], label='D_H')
# plt.plot(t, D[:, 0, 3], label='D_ICU')
# # plt.plot(t, np.sum(sol, axis=1), label='N')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.legend()
# plt.show()
#
# plt.plot(t, I[:, 0, 0], label='Infected Asymptomatic')
# plt.plot(t, np.sum(I[:, 0, 1:], axis=1), label='Infected Symptomatic')
# # plt.plot(t, sol[:, 3], label='I_GP')
# # plt.plot(t, sol[:, 4], label='I_H')
# # plt.plot(t, sol[:, 5], label='I_ICU')
# plt.plot(t, np.sum(I[:, 0], axis=1), label='Total Infected')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Young')
# plt.legend()
# plt.show()
#
# plt.plot(t, I[:, 1, 0], label='Infected Asymptomatic')
# plt.plot(t, np.sum(I[:, 1, 1:], axis=1), label='Infected Symptomatic')
# # plt.plot(t, sol[:, 3], label='I_GP')
# # plt.plot(t, sol[:, 4], label='I_H')
# # plt.plot(t, sol[:, 5], label='I_ICU')
# plt.plot(t, np.sum(I[:, 1], axis=1), label='Total Infected')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Old')
# plt.legend()
# plt.show()
#
# plt.plot(t, I[:, 0, 1], label='I_GP')
# plt.plot(t, I[:, 0, 2], label='I_H')
# plt.plot(t, I[:, 0, 3], label='I_ICU')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Young')
# plt.legend()
# plt.show()
#
# plt.plot(t, I[:, 1, 1], label='I_GP')
# plt.plot(t, I[:, 1, 2], label='I_H')
# plt.plot(t, I[:, 1, 3], label='I_ICU')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Old')
# plt.legend()
# plt.show()
#
# plt.plot(t, R[:, 0, 1], label='Recovered Asymptomatic')
# plt.plot(t, R[:, 0, 2], label='Recovered Symptomatic')
# plt.plot(t, np.sum(R[:, 0, 3:], axis=1), label='Recovered Serious')
# plt.plot(t, np.sum(R[:, 0], axis=1), label='Recovered Total')
# # plt.plot(t, sol[:, 8], label='R_H')
# # plt.plot(t, sol[:, 9], label='R_ICU')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Young')
# plt.legend()
# plt.show()
#
# plt.plot(t, R[:, 1, 1], label='Recovered Asymptomatic')
# plt.plot(t, R[:, 1, 2], label='Recovered Symptomatic')
# plt.plot(t, np.sum(R[:, 1, 3:], axis=1), label='Recovered Serious')
# plt.plot(t, np.sum(R[:, 1], axis=1), label='Recovered Total')
# # plt.plot(t, sol[:, 8], label='R_H')
# # plt.plot(t, sol[:, 9], label='R_ICU')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Old')
# plt.legend()
# plt.show()
#
# plt.plot(t, np.sum(D[:, 0], axis=1), label='Deceased')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Young')
# plt.legend()
# plt.show()
#
# plt.plot(t, np.sum(D[:, 1], axis=1), label='Deceased')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Old')
# plt.legend()
# plt.show()
#
# plt.plot(t, S[:, 0] + E[:, 0] + np.sum(I[:, 0], axis=1) + np.sum(R[:, 0], axis=1) + np.sum(D[:, 0], axis=1), label='N')
# plt.xlabel('Time (days)')
# plt.ylabel('Cases')
# plt.title('Young')
# plt.legend()
# plt.show()
