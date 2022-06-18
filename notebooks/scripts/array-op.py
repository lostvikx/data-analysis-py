# %% [markdown]
# # Array-Oriented Programming
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
test_mesh = np.arange(0,5)
# Both xs and ys are (1000, 1000) 2D matrices
xs, ys = np.meshgrid(test_mesh, test_mesh)

# %% [markdown]
# Now the pairing of the corresponding element in each matrix gives us the x and y coords of a point on the grid.

#%%
plt.plot(xs, ys, marker=".", linestyle="none")
plt.show()
# %% [markdown]
# ### Test Mathematical Functions
# Suppose we want to evaluate the math function `sqrt(x^2 + y^2)` across a grid of values.

# %%
# 1000 equally-spaced points
points = np.arange(-5,5,0.01)
x, y = np.meshgrid(points, points)
# %%
z = np.sqrt(x**2 + y**2)

# %%
plt.imshow(z, extent=[-5,5,-5,5], cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()

# %% [markdown]
# ### Expressing Conditionals

# %% [markdown]
# Suppose we have a boolean array and two arrays with floats.

# %%
xarr = (np.random.standard_normal(5) * 2).round(2)
yarr = (np.random.standard_normal(5) * 2).round(2)
cond = np.array([True, False, True, True, False])

# %% [markdown]
# We wanted to take a value from `xarr` whenever the corresponding value in cond is True, and otherwise take the value from `yarr`.

# %%
cond_res = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
cond_res

# %% [markdown]
# The above code as a few problems:
# * It's slow
# * Doesn't work for multi-dimentional arrays

# %%
np.where(cond, xarr, yarr)

# %% [markdown]
# We can even use scalar values, instead of arrays:

# %%
# Replace all negative values with 0 and all positive values with 1
test_matrix = np.random.standard_normal((4, 4)).round(2)
test_matrix

#%%
mat_cond = test_matrix < 0
np.where(mat_cond, 0, 1)

# %% [markdown]
# We can use both scalar and arrays:

# %%
# Only replace all negative values with 0
np.where(mat_cond, 0, test_matrix)

# %% [markdown]
# ### Mathematical and Statistical Methods

# %% [markdown]
# We can use aggregation functions like `mean` or `std`

# %%
math_arr = np.random.standard_normal((5, 4))

# %%
# Instance method
math_arr.mean()

# %%
# Top-level method
np.mean(math_arr)

# %%
math_arr.sum()

# %% [markdown]
# Some methods take `axis` argument

# %%
# Compute mean across rows
print(math_arr.mean(axis=0))
# Comptute mean across columns
print(math_arr.mean(axis=1))

# %% [markdown]
# Other methods like `cumsum` or `cumprod` do not aggregate, they produce an array of intermediate results.

# %%
test_cum = np.arange(1, 5)
test_cum.cumprod()

# %%
matrix_cum = np.arange(0, 9).reshape(3, 3)
matrix_cum

# %%
# Creates a 1D array of cummulative sums
matrix_cum.cumsum()

# %%
# Compute mean for values in different rows [|]
matrix_cum.cumsum(axis=0)

# %%
# Compute mean for values in different columns [-]
matrix_cum.cumsum(axis=1)

# %%

