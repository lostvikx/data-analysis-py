#%% [markdown]
# # Descriptive Statistics
# 
# `pandas` object come with built-in set of common mathematical and statical methods. Most fall in the category of reduction or summary statistics.

# %%
import pandas as pd
import numpy as np

# %%
df=pd.DataFrame(
  np.random.standard_normal((4,2)).round(2)*8,
  index=list("abcd"),
  columns=["one", "two"]
)
df

# %%
df.loc["a", "two"] = np.nan
df.loc["c"] = np.nan
df

# %%
df.sum()

# %%
# Sum across columns:
df.sum(axis=1)

# %% [markdown]
# When the entire row contains `nan` values, the sum is 0.

# %%
df.sum(axis=1, skipna=False)

# %% [markdown]
# When using `skipna` param, any `nan` value in a row will result in `nan`.
#
# Some aggregation methods, like `mean`, require at least one non-NA value:

# %%
df.mean(axis=1)

# %%
df.idxmax()

# %%
df.cumsum()

# %%
df.describe()

# %%
pd.Series(["a", "a", "b", "c"]*4).describe()

# %%
df.quantile()

# %%
df.mad()

# %%
df.std()

# %%

