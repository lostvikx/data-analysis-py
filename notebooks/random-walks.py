# %% [markdown]
# # Simple Random Walk

#%%
import numpy as np
import random
import matplotlib.pyplot as plt

# %%
# Simple Python implementation
pos = 0
walk = []
n_steps = 1000

for _ in range(n_steps):
  # eg: like a coin toss, only +1 or -1
  step = 1 if random.randint(0,1) else -1
  pos += step
  walk.append(pos)

plt.plot(walk[:100])
plt.title(f"Avg pos: {np.mean(walk)}")
plt.show()

# %%
# Numpy implementation
draws = np.random.randint(0,2, size=n_steps)
steps = np.where(draws, 1, -1)
np_walk = steps.cumsum()

plt.plot(np_walk[:100])
plt.title(f"Avg pos: {np.mean(np_walk)}")
plt.show()

# %%
# Simple stats
np_walk.min()

# %%
np_walk.max()

# %%
# Let's say we want to find out the first crossing time of the walk exceeding 10 steps (+ or -)
(np.abs(np_walk) == 10).argmax()

# %% [markdown]
# **Note**: `argmax` isn't the most efficient method of finding the first True value.

# ### Power of array-oriented programming

# The above was for only one walk of 1000 steps, let's simulate 5000 walks (each has 1000 steps):

# %%
n_walks = 5000
multi_draws = np.random.randint(0,2, size=(n_walks, n_steps))
# multi_draws.shape
multi_steps = np.where(multi_draws, 1, -1)
multi_walk = multi_steps.cumsum(axis=1)

multi_walk

# %%
multi_walk.max()

# %%
multi_walk.min()

# %% [markdown]
# **Note**: not all walks hit 30 steps

# Does any walk go beyond 30 steps, if yes return `True` else `False`.

# %%
# First part creates boolean arrays for each walk, next creates a 1D boolean array.
cross30 = (np.abs(multi_walk) == 30).any(axis=1)
cross30

# %%
# Total number of walks that crossed the 30-step threshold
cross30.sum()

# %% [markdown]
# We can use the above boolean array to slice out the walks that have crossed 30 steps.

# %%
crossing_time = (np.abs(multi_walk[cross30]) == 30).argmax(axis=1)
crossing_time

# %%
# Verify the first case of crossing 30:
multi_walk[cross30.argmax(),crossing_time[0]]

# %% [markdown]
# Calculate some stats for crossing time:

# %%
# Fastest crossing time:
crossing_time.min()

# %%
# Slowest crossing time:
crossing_time.max()

# %%
# Mean crossing time:
crossing_time.mean()
