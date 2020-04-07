import numpy as np
import matplotlib.pyplot as plt

from seir.model import NInfectiousModel
from seir.wrapper import MultiPopWrapper
from seir.utils import plot_solution

q_ii = np.array([
    [0, 0, 0, 0],
    [0, 1 / 5, 0, 0],
    [0, -1 / 5, 1 / 6, 0],
    [0, 0, -1 / 6, 0]
])
# q_ii = np.repeat(np.expand_dims(q_ii, 0), 2, axis=0)

infectious_func = lambda t: 1 if t <= 17 else 0.2 if 17 < t < 38 else 1
imported_func = lambda t: [[0, 0.75 * 9 * np.exp(0.11*t), 0, 0], [0, 0.25 * 9 * np.exp(0.11*t), 5, 0]] if t < 16 else 0

model = MultiPopWrapper(
    pop_categories={'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                    'sex': ['male', 'female'],
                    'risk': ['high', 'low']
                    },
    inf_labels=['AS', 'GP', 'H', 'ICU'],
    alpha=[0.179, 0.821, 0, 0.],
    t_inc=5.1,
    q_se=[0.4163, 0.8326, 0, 0],
    q_ii=q_ii,
    q_ir=[1/10, 1/2.3, 1/8, 1/10],
    q_id=[0, 0, 0, 1/5],
    delta=[0, 0.012, 0.05, 0],
    beta=[0, 0, 0, 0.609],
    infectious_func=infectious_func,
    imported_func=imported_func,
    extend_vars=True
)

print(model.pop_categories)
print(model.nb_groups)
print(model.pop_labels)
print(model.pop_label_to_idx)

init_vectors = {
    's_0': [27000000, 8000000]
}
t = np.linspace(0, 300, 10000)
solution = model.solve(init_vectors, t, to_csv=True, fp='data/solution.csv')

print(model.r_0_eff)
print(model.r_0)

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


