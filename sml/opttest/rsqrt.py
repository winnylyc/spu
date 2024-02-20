import jax
import jax.numpy as jnp

# def rsqrt_unopt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     return X / jnp.sqrt(norms)[:, jnp.newaxis]

# def rsqrt_opt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     norms = norms.astype(jnp.float32)
#     return X * jax.lax.rsqrt(norms)[:, jnp.newaxis]

# def rsqrt_unopt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     return X / (X * jnp.sqrt(norms)[:, jnp.newaxis])

# def rsqrt_opt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     norms = norms.astype(jnp.float32)
#     return X / X * jax.lax.rsqrt(norms)[:, jnp.newaxis]

# def rsqrt_unopt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     return X / (X[:, jnp.newaxis] * jnp.sqrt(norms)[:, jnp.newaxis])

# def rsqrt_opt(X):
#     norms = jnp.einsum("ij,ij->i", X, X)
#     norms = norms.astype(jnp.float32)
#     return X / X[:, jnp.newaxis] * jax.lax.rsqrt(norms)[:, jnp.newaxis]

# def rsqrt_unopt(X):
#     norms = (X * X).sum(axis=1)
#     norms = norms.astype(jnp.float32)
#     return X / jnp.sqrt(X[:, jnp.newaxis])

# def rsqrt_opt(X):
#     norms = (X * X).sum(axis=1)
#     norms = norms.astype(jnp.float32)
#     return X * jax.lax.rsqrt(X[:, jnp.newaxis])

# def rsqrt_unopt(X):
#     X = X.astype(jnp.float32)
#     return X / jnp.sqrt(X)

# def rsqrt_opt(X):
#     X = X.astype(jnp.float32)
#     return X * jax.lax.rsqrt(X)

# def rsqrt_unopt(X):
#     X = X.astype(jnp.float32)
#     return X / (X * jnp.sqrt(X))

# def rsqrt_opt(X):
#     X = X.astype(jnp.float32)
#     return X / X * jax.lax.rsqrt(X)

def rsqrt_unopt(X):
    # X = X.astype(jnp.float32)
    return X / (X * jnp.sqrt(X)[:, jnp.newaxis])

def rsqrt_opt(X):
    X = X.astype(jnp.float32)
    return X / X * jax.lax.rsqrt(X[:, jnp.newaxis])