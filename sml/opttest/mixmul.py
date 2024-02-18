import jax
import jax.numpy as jnp

def mixmul_unopt(X, Y):
    return X * Y

def mixmul_opt(X, Y):
    return X * Y