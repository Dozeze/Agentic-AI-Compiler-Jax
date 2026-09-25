#ONLY ALLOWED IMPORTS

import jax
import jax.numpy as jnp

# Data and max iterations
max_iter = 1000
data_points = 20
data_x = jnp.linspace(0, 20, data_points)
data_y = jnp.exp(jnp.linspace(0, 10, data_points)) * 1.03 + 1.32
eta = 0.1

# Functions
def func(x, c):
    complicated_thing = 1.03
    A = 13.03
    B = 1032.02
    C = 132.3
    D = A + B
    E = A + C + B
    F = D + E + A
    return 3 * jnp.exp(x) + c * 4 * complicated_thing + E / F

def loss(x, y, f, c):
    loss_val = 0.5 * jnp.mean(f(x, c) - y)
    return loss_val

# Algorithm to run:

c = 100

for it in range(max_iter):
    grad = jax.grad(func(data_x, data_y, func, c), argnums = 4)
    c -= eta * grad
    

