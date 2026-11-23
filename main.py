import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

t_final = 2
nr_t_steps = 200000
times = np.linspace(0, t_final, nr_t_steps)
t_step = times[1] - times[0]

x_final = 20
nr_x_steps = 200
xs = np.linspace(0, x_final, nr_x_steps)
x_step = xs[1]-xs[0]

psi = np.zeros(len(xs), dtype=complex)

for i, x in enumerate(xs):
    if 8 < x < 9:
        psi[i] = np.sin(np.pi*(x-8))
    if 10 < x < 11:
        psi[i] = np.sin(np.pi*(x-10))

prob = np.abs(psi)**2
total = np.sum(prob)
psi = psi/np.sqrt(total)
prob = np.abs(psi)**2

def second_der(f, step):
    result = np.zeros_like(f, dtype=complex)
    result[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2])/step**2
    return result

fig, ax = plt.subplots()
ax.set_ylim(0, 0.2)
plot, = ax.plot(xs, prob)

def se_rhs(psi):
    return 1j * second_der(psi, x_step)

psi_evolution = np.zeros((len(times), len(xs)), dtype=complex)

for i, t in enumerate(times):
    prob = np.absolute(psi)**2
    local_t = 0.001 * t_step / np.maximum(prob**2, 0.0001)
    k1 = local_t * se_rhs(psi)
    k2 = local_t * se_rhs(psi + 0.5*k1)
    k3 = local_t * se_rhs(psi + 0.5*k2)
    k4 = local_t * se_rhs(psi + k3)

    psi = psi + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    prob = np.absolute(psi)**2
    total = np.sum(prob)
    psi = psi/np.sqrt(total)
    psi_evolution[i] = psi

# for i, t in enumerate(times):
#     ddx2 = second_der(psi, x_step)
#     ddt = 1j * ddx2
#     psi = psi + t_step * ddt
#     prob = np.absolute(psi)**2
#     total = np.sum(prob)
#     psi = psi/np.sqrt(total)
#     psi_evolution[i] = psi

def update(frame):
    psi_at_t = psi_evolution[frame]
    prob = np.absolute(psi_at_t)**2
    plot.set_data(xs, prob)
    return plot

ani = FuncAnimation(fig, update, frames=range(0, len(times), 1000))
plt.show()