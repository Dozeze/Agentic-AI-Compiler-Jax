import jax


def softmax(x):
    return jax.nn.softmax(x, axis=-1)           # control: XLA already compiles the reference well
