import numpy as np
import matplotlib.pyplot as plt

from seir.model import NInfectiousModel
from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution

infectious_func = lambda t: 1 if t <= 17 else 0.2 if 17 < t < 38 else 1
# imported_func = lambda t: [[0, 0.75 * 9 * np.exp(0.11*t), 0, 0], [0, 0.25 * 9 * np.exp(0.11*t), 5, 0]] if t < 16 else 0
imported_func = None

model = MultiPopWrapper(
    pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                    'sex': ['male', 'female'],
                    'risk': ['low', 'high']
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
    q_se=[0.4163, 0.8326, 0.8326, 0, 0, 0],
    q_ii=[
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1/2.3, 0, 0, 0],
        [0, 0, -1/2.3, 1/2.7, 0, 0],
        [0, 0, 0, -1/2.7, 1/6, 0],
        [0, 0, 0, 0, -1/6, 0]
    ],
    q_ir=[1 / 10, 1 / 2.3, 0, 0, 1 / 8, 1 / 10],
    q_id=[0, 0, 0, 0, 0, 1 / 5],
    rho_delta={'20-29': [0, 0, 1, 1, 0.05, 0],
               '60-69': [0, 0, 1, 1, 0.274, 0]},
    rho_beta={'20-29': [0, 0, 0, 0, 0, 0.609],
              '60-69': [0, 0, 0, 0, 0, 0.589]},
    infectious_func=infectious_func,
    imported_func=imported_func,
    extend_vars=True
)

init_vectors = {
    's_0': {'20-29_male_low': 27000000,
            '60-69_male_low': 8000000},
    'i_0': {'20-29_male_low': [0, 100, 0, 0, 0, 0]}
}

t = np.linspace(0, 300, 1501)
solution = model.solve(init_vectors, t, to_csv=True, fp='data/solution.csv')


print(model.r_0)

s_t, e_t, i_t, r_t, d_t = solution

# plot all figures
fig, axes = plot_solution(solution, t)

# for row in axes:
#     for ax in row:
#         ax.set_xlim((0, 50))
#         ax.set_ylim((0, 50))

plt.show()

# plot young
# fig, axes = plot_solution(solution, t, 0)
# plt.show()
#
# # plot old
# fig, axes = plot_solution(solution, t, 1)
# plt.show()
