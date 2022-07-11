#!usr/bin/env python

# %% [markdown]
# # Data Cleaning
#
# Data cleaning is one of the most important and time consuming process of any data analysis project. Such tasks are often reported to take up 80% or more of an analyst's time.

# %% [markdown]
# ### Handling Missing Data

# %%
import pandas as pd
import numpy as np

# %%
ser1 = pd.Series([1.2,4.1,None,3.8,np.nan])
ser1

# %%
ser1.isna()

# %%
ser1.dropna()

# %%
ser1.fillna(value=0)

# %%
ser1.fillna(method="ffill")

# %%
ser1.fillna(value=ser1.mean()).round(1)

# %%
# filter only nan values
ser1[ser1.isna()]

# %%
# same as dropna
ser1[ser1.notna()]

# %%
df1 = pd.DataFrame([[1.2,2.1,4.6],[np.nan,None,np.nan],[6.2,np.nan,9.1],[np.nan,8.2,np.nan]])
df1

# %%
# Drops rows by default
df1.dropna()

# %%
df1.dropna(axis=1)

# %%
df1.dropna(how="all")

# %%
df1.dropna(axis=1, how="all")

# %%
df2 = pd.DataFrame(np.random.standard_normal((7,3)))
df2.iloc[:2,2] = np.nan
df2.iloc[:4,1] = np.nan
df2

# %%
df2.dropna()

# %% [markdown]
# We can specify a threshold, max number of rows or cols to be dropped:

# %%
# drop 2 rows max
df2.dropna(thresh=2)

# %% [markdown]
# ### Filling Missing Data
#
# Rather than filtering out missing data (and potentially discarding other data along with it), you may want to fill in the “holes” in any number of ways.

# %%
# Don't mutate the original
# df2.columns = list("abc")
# df2

# Use this instead
df3 = df2.set_axis(list("abc"),axis=1).round(2)
df3

# %% [markdown]
# Fill with column specific values:

# %%
df3.fillna({"a":0.2,"b":0.4,"c":0.8})

# %%
# Fill backwards:
df3.fillna(method="bfill")

# %%
# Fill across columns
df3.fillna(method="ffill", axis=1)

# %%
# Can also have a threshold:
df3.fillna(method="ffill", axis=1, limit=1)

# %%
df3.fillna(df3.mean(axis=0)).round(2)

# %% [markdown]
# ### Data Transformation

# %%

