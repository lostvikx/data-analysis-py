#%% [markdown]
# # Data Wrangling

# %%
import pandas as pd
import numpy as np

# %%
ser1 = pd.Series(np.random.uniform(size=9).round(2),index=[list("aaabbccdd"),[1,2,3,2,3,1,2,1,3]])
ser1

# %%
ser1.index

# %% [markdown]
# ### Selecting data in a MultiIndex Series

# %%
ser1.loc["a"]

# %%
ser1.loc[["a","d"]]

# %%
ser1.loc["a", 1]
#OR
ser1["a"][1]

# %%
ser1.loc["b":"d"]

# %%
# Select only the inner level:
ser1.loc[:, 2]

# %% [markdown]
# Note: Hierarchical indexing can be used to form a pivot table.

# %%
ser1.unstack()

# %%
# Inverse of unstack is:
ser1.unstack().stack()

# %% [markdown]
# Note: Either axis can have MultiIndex or hierarchical index.

# %%
df1 = pd.DataFrame(
  np.random.standard_normal(size=(4,3)).round(3),
  index=[list("aabb"),[1,2,1,2]],
  columns=[["Ohio", "Ohio", "Texas"],["blue", "red", "white"]]
)
df1

# %%
df1.index.names = ["key1","key2"]
df1.columns.names = ["state","color"]
df1

# %%
# check the levels of index or columns
df1.index.nlevels
df1.columns.nlevels

# %%
df1["Ohio"].loc["b"]

# %%
pd.MultiIndex.from_arrays(
  [["Ohio","Ohio","Texas"],["blue","red","white"]],
  names=["state","color"]
)

#%% [markdown]
# Note: Sometimes we may require to swap or change the order of levels in a MultiIndex

# %%
df1.swaplevel(0,1)
# OR
df1.swaplevel("key1","key2")

#%% [markdown]
# Note: The data remains unaltered
#
# `sort_index` by defaults sorts data lexicographically using all index levels, but we can specify a single level or a subset of levels.

# %%
# level= Name (str) || Number (int) || List
# Example: level="key2" || level=1
df1.sort_index(level=1)

# %%
df1.swaplevel(0,1).sort_index(level=0)

# %%
df1.groupby(level="key2").sum()

# %%
df1.groupby(level="color",axis=1).sum()
