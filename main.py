import numpy as np
import matplotlib.pyplot as plt

from seir.model import NInfectiousModel
from seir.utils import plot_solution

q_ii = np.array([
    [0, 0, 0, 0],
    [0, 1 / 5, 0, 0],
    [0, -1 / 5, 1 / 6, 0],
    [0, 0, -1 / 6, 0]
])
q_ii = np.repeat(np.expand_dims(q_ii, 0), 2, axis=0)

model = NInfectiousModel(
    nb_groups=2,
    nb_infectious=4,
    alpha=[[0.179, 0.821, 0, 0.], [0.179, 0.821, 0, 0.]],
    t_inc=5.1,
    q_se=[0.4163, 0.8326, 0, 0],
    q_ii=q_ii,
    q_ir=[[1/10, 1/2.3, 1/8, 1/10], [1/10, 1/2.3, 1/8, 1/10]],
    q_id=[[0, 0, 0, 1/5], [0, 0, 0, 1/5]],
    delta=[[0, 0.012, 0.05, 0], [0, 0.166, 0.274, 0]],
    beta=[[0, 0, 0, 0.609], [0, 0, 0, 0.589]]
)

init_vectors = {
    's_0': [27000000, 8000000],
    'e_0': [240, 0],
    'i_0': [[0, 100, 0, 0], [0, 0, 0, 0]]
}
t = np.linspace(0, 300, 10000)
infectious_func = lambda t: 1 if t <= 7 else 0.2 if 7 < t < 28 else 1
solution = model.solve(init_vectors, t, infectious_func, to_csv=True, fp='data/solution.csv')

# plot all figures
fig, axes = plot_solution(solution, t)
plt.show()

# plot young
# fig, axes = plot_solution(solution, t, 0)
# plt.show()
#
# # plot old
# fig, axes = plot_solution(solution, t, 1)
# plt.show()


