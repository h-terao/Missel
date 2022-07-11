import jax
import jax.numpy as jnp
import jax.random as jr


def interleave(xy, batch):
    nu = len(xy) - 1

    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [jnp.concatenate(v) for v in xy]


def jittable(xy, batch):
    nu = len(xy) - 1
    xy = [[x[::-1] for x in reversed(jnp.array_split(v[::-1], nu + 1))] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [jnp.concatenate(v) for v in xy]


xy = [jr.uniform(key, [8]) for key in jr.split(jr.PRNGKey(0), 5)]
print(xy)
print(interleave(xy, 8))
print("+++")
print(jax.jit(jittable)(xy, 8))
print(jax.jit(jittable)(jax.jit(jittable)(xy, 8), 8))
