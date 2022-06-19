# %% [markdown]
# # Linear Algebra

# %%
import numpy as np

# %%
def rand_arr(shape=5, times=1):
  return np.random.standard_normal(shape) * times

# %%
a = rand_arr((2, 3))
b = rand_arr((3, 2))

# %% [markdown]
# ![Dot product](imgs/dot-product-matrix.jpg)

# %%
np.dot(a, b)
# Or
a.dot(b)

# %%
a * np.ones(3)

# %%
c=rand_arr((2,2))
d=rand_arr((2,2))

# %% [markdown]
# **Note**: Element-wise `*` multiplication is not equal `dot` product of two matrices:

# %%
print(c*d == c.dot(d))

# %%
x = np.ceil(rand_arr((3, 3)))
y = np.ceil(rand_arr((3, 3)))

# %% [markdown]
# ![How to inverse a matrix](imgs/inverse-matrix.jpg)

# %%
mat_x = x.T @ x
inv_x = np.linalg.inv(mat_x)

mat_x @ inv_x

# %%
y.dot(y.T) == y.T.dot(y)

# %%
[fn for fn in dir(np.linalg) if fn[0] != "_"]

# %%
help(np.linalg.inv)

# %%
# sum of the diagnal of a matrix
np.trace(y)
