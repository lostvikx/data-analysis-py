# %% [markdown]
# # Data Visualization & Plotting

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

# %%
# defaults to subplots(1,1)
fig, ax = plt.subplots()
# plot y using x axis as 0 to n-1
ax.plot(
  np.random.standard_normal(30).cumsum(),
  color="black",
  linestyle="--",
  marker="o"
);

# %%
fig, ax = plt.subplots()
d = np.random.standard_normal(30).cumsum()

ax.plot(d,"o--",label="default"); # linear interpolation by default
ax.plot(d,"-",label="steps-post",drawstyle="steps-post");

ax.set_title("Change drawstyle")
ax.legend();

# %% [markdown]
# **Note**: Calling the `legend` method is mandatory to create a legend, regardless of whether label option is passed while plotting.

# %% [markdown]
# ### Ticks & Labels
#
# Simple plot decoration methods: `xlim`, `xticks`, `xticklabels`, also the `y` counterparts.
#
# We can call the settter and getter methods on them:

# %%
fig, ax = plt.subplots()
ax.plot(np.random.standard_normal(1000).cumsum(),"black");
# X axis
ax.set_xticks([0,250,500,750,1000]);
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May"],rotation=30);
ax.set_xlabel("Month");

# Y axis
ax.set(
  title="Monthly % Change in Sales",
  ylabel="% Change"
);

# %% [markdown]
# Note: Use the `set` method to simplify things.

# %%
fig, ax = plt.subplots()

ax.plot(np.random.standard_normal(100),"-",label="one");
ax.plot(np.random.standard_normal(100),"--",label="two");
ax.plot(np.random.standard_normal(100),".-",label="__nolegend__");

ax.legend();

# %% [markdown]
# Note: To exclude one or more plots from legend, use `label=__nolegend__` in label, to be more explicit.

# %% [markdown]
# ### Annotations & Drawings on Subplot

# %%
fig, ax = plt.subplots()
sp500_data = pd.read_csv("examples/spx.csv",index_col=0,parse_dates=True)
spx = sp500_data["SPX"]

spx.plot(ax=ax,color="black")

crisis_data = [
  (datetime(2007,10,11), "Peak of bull market"),
  (datetime(2008,3,12), "Bear Stearns fails"),
  (datetime(2008,9,15), "Lehman Brothers bankruptcy")
]

for date, label in crisis_data:
  ax.annotate(
    label, 
    xy=(date, spx.loc[date]+75),
    xytext=(date, spx.loc[date]+225),
    arrowprops=dict(arrowstyle="->"),
    horizontalalignment="left",
    verticalalignment="top"
  );

ax.set(
  xlim=["2007-1-1","2011-1-1"],
  ylim=[600,1800],
  title="Important dates in the 2008 financial crisis",
  ylabel="S&P 500"
);

fig.savefig("out/2008_financial_crisis.png",dpi=400)

# %%
fig, ax = plt.subplots()

rect = plt.Rectangle((0.2,0.2),0.4,0.1,alpha=0.4)
circ = plt.Circle((0.7,0.6),0.2,alpha=0.3,color="green")
pgon = plt.Polygon([(0.2,0.4),(0.4,0.65),(0.1,0.8)],alpha=0.4,color="yellow")

ax.add_patch(rect);
ax.add_patch(circ);
ax.add_patch(pgon);

# %% [markdown]
# ### Save Plots to Files

# %%
fig.savefig("out/patch_shapes.svg")

# %% [markdown]
# ### Configuration
#
# Use the `rc` method to change the defaults on figure, plot, or colors.

# %%
plt.rc("figure", figsize=(10,10))

# %%
# Default configuration:
plt.rcParams

# %% [markdown]
# First argument of `rc`: `figure`, `axes`, `xticks`, `yticks`, `grid`, or `legend`.

# %%
plt.rc("figure", figsize=(6,4))
plt.rc("font",family="monospace",weight="300",size=10)

fig, ax = plt.subplots()
dt = pd.Series(np.random.standard_normal(40).cumsum())
ax.plot(dt,"o--",label="one");
point = 20
ax.annotate(
  "random point",
  xy=(point, dt[point]),
  xytext=(point+2,dt[point]+2),
  arrowprops=dict(arrowstyle="->")
)

ax.legend();

# %% [markdown]
# ## Plotting with Seaborn
#
# `matplotlib` can be a bit low-level, we can use a higher level plotting library called seaborn. A DataFrame can have a lot of columns, seaborn provides a lot of common plotting options.

# %%
import seaborn as sns
