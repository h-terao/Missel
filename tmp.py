import jax
import jax.numpy as jnp


def interleave_fn(batch):
    @jax.jit
    def f(xy):
        nu = len(xy) - 1
        xy = [jnp.array_split(v, nu) for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return xy

    return f


xy = [jnp.zeros(10) for _ in range(5)]
print(interleave_fn(10)(xy))
