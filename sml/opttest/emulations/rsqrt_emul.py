import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing

import sml.utils.emulation as emulation
from sml.opttest.rsqrt import rsqrt_unopt, rsqrt_opt

def emul_rsqrt_unopt():
    def func(X):
        return rsqrt_unopt(X)

    X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

    X = emulator.seal(X)
    result = emulator.run(func)(X)
    print(result)

def emul_rsqrt_opt():
    def func(X):
        return rsqrt_opt(X)

    X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

    X = emulator.seal(X)
    result = emulator.run(func)(X)
    print(result)

if __name__ == "__main__":
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC,
            emulation.Mode.MULTIPROCESS,
            bandwidth=300,
            latency=20,
        )
        emulator.up()
        emul_rsqrt_unopt()
        emul_rsqrt_opt()
    finally:
        emulator.down()