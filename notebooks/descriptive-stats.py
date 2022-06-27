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

# %% [markdown]
# ### Correlation & Covariance

# %%
price = pd.read_pickle("examples/yahoo_price.pkl")
vol = pd.read_pickle("examples/yahoo_volume.pkl")

# %%
price.tail()

# %%
returns = price.pct_change()

# %%
returns.tail()

# %%
returns["IBM"].corr(returns["MSFT"])

# %%
returns["AAPL"].cov(returns["GOOG"])

# %% [markdown]
# **Note**: If we use the `corrwith` method, it computes the corr of each column with a Series passed as the argument.

# %%
returns.corrwith(returns["IBM"])

# %% [markdown]
# We can even pass a DataFrame as the argument, it computes the corr for the matching column names:

# %%
returns.corrwith(vol, axis=0)

# %%
rep_ser = pd.Series(["a", "c", "a", "c", "b", "b", "c", "b", "d"])
rep_ser.unique()

# %%
rep_ser.value_counts()

# %%
mask = rep_ser.isin(["a", "b"])

# %%
# Filtering the data:
rep_ser[mask]

# %%
uni_vals = pd.Series(["c", "a", "b"])
indices = pd.Index(uni_vals).get_indexer(rep_ser)

# %%
# -1 because uni_vals doesn't contain "d"
indices

# %%
dt = pd.DataFrame({"a":[4,4,1,2,3],"b":[2,2,1,3,5],"c":[1,2,1,1,4]})
dt

# %% [markdown]
# Compute value counts of a single column:

# %%
dt["a"].value_counts().sort_index()

# %% [markdown]
# Compute value counts for every column with the apply method:

# %%
dt.apply(pd.value_counts).fillna(0)

# %% [markdown]
# There is also a built-in `value_counts` method, but it computes the counts considering each row of the DataFrame.

# %%
dt_1 = pd.DataFrame({
  "a": [1,1,0,0,2],
  "b": [1,1,2,2,0],
  "c": [1,1,1,1,2]
})
dt_1

# %%
# The index represents the unique rows as a hierarchical index:
dt_1.value_counts()
