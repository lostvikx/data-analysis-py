# %% [markdown]
# # Data Visualization & Plotting

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# %matplotlib inline

# %%
plt.plot(np.arange(10))
plt.show()

# %% [markdown]
# ### Figures & Subplots
#
# Plots in `matplotlib` reside in the `Figure` object, here is how to create one:

# %%
fig = plt.figure()
# We need to create a subplot in the figure, figure is just like an empty canvas.
ax1 = fig.add_subplot(2,2,1) # 2x2, 1st subplot
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)

# simple line plot
ax3.plot(
  np.random.standard_normal(50).cumsum(),
  color="black",
  linestyle="dashed"
);

# histogram
ax1.hist(
  np.random.standard_normal(100),
  bins=20,
  color="black",
  alpha=0.25
);

# scatter plot
ax2.scatter(
  np.arange(30),
  np.arange(30) + 3 * np.random.standard_normal(30)
);

# %% [markdown]
# A more convenient way to create subplots: 

# %%
fig, axes = plt.subplots(2,2)
# numpy array containing created subplot objects
axes

# %%
fig, axes = plt.subplots(2,2,sharex=True,sharey=True)

for i in range(2):
  for j in range(2):
    axes[i,j].hist(
      np.random.standard_normal(500),
      bins=50,
      color="black",
      alpha=0.5
    );

# Sets the padding of each plots to 0
fig.subplots_adjust(wspace=0,hspace=0)

# %% [markdown]
# ### Colors, Markers, Line Styles

# %%
"When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you."
