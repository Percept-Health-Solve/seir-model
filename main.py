import numpy as np
import matplotlib.pyplot as plt
from seir.model import NInfectiousModel

q_ii = np.array([
    [1 / 10, 0, 0, 0],
    [0, 1 / 5, 0, 0],
    [-1 / 10, -1 / 5, 1 / 4, -1E-20],
    [0, 0, -1 / 4, 1E-20]
])
q_ii = np.repeat(np.expand_dims(q_ii, 0), 2, axis=0)

model = NInfectiousModel(
    nb_groups=2,
    nb_infectious=4,
    alpha=[[0.6, 0.4, 0, 0.], [0.4, 0.3, 0.3, 0.]],
    t_inc=5,
    q_se=[0.1, 0.9, 0, 0],
    q_ii=q_ii,
    q_ir=[[1/10, 1/5, 1/7, 1/14], [1/10, 1/5, 1/7, 1/14]],
    q_id=[[1/10, 1/5, 1/7, 1/14], [1/10, 1/5, 1/7, 1/14]],
    beta=[[0.0, 0.0, 0.01, 0.1], [0.0, 0.093, 0.093, 0.093]],
    delta=[[0.0, 0.012, 0.05, 0], [0.1, 0.273, 0.709, 0]]
)

init_vectors = {
    's_0': [25000000, 5000000],
    'i_0': [[0, 100, 0, 0], [0, 0, 0, 0]]
}

t = np.linspace(0, 300, 10000)

solution = model.solve(init_vectors, t)

plt.plot(t, np.sum(solution['d_t'], axis=(2)))
plt.show()

