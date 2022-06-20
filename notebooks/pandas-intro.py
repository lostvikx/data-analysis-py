#%% [markdown]
# # Pandas Intro

#%%
import pandas as pd
import numpy as np

#%% [markdown]
# ### Built-in Data Structures

# A `Series` has `array` and `index` attributes

# %%
test_obj = pd.Series([1, 2, -1, -8], index=["a", "b", "c", "d"])
test_obj

# %%
# This is a PandasArray, it kinda wraps a NumPy array
test_obj.array

# %%
test_obj.index

# %%
test_obj["c"]

# %%
test_obj["a"] = 3

# %%
test_obj[["a", "d"]]

# %% [markdown]
# **Note**: Using NumPy-like operations, such as filtering, scalar multiplication, or applying math operations, will preserve the index-value link:

#%%
test_obj[test_obj > 0]

# %%
test_obj * 2

# %%
np.exp(test_obj)

# %%
s_data = {"Assam": 2000, "Delhi": 7000, "Bengal": 5000, "Kerla": 4000}
obj0 = pd.Series(s_data)
obj0

# %%
# Convert it back to dict:
obj0.to_dict()

# %% [markdown]
# **Note**: When only passing a dict in the `Series`, result will follow the key insertion order, we can override this by passing an `index=` param:

# %%
states = ["Kerla", "Goa", "Delhi", "Assam"]
obj1 = pd.Series(s_data, index=states)
obj1

# %% [markdown]
# We can see one `NaN` value in our Series, `NaN` is Not a Number in pandas that marks missing data values.

# %%
obj1.isna()
# OR
# obj1.notna() # to get the opposite

# %% [markdown]
# Series automatically aligns by index label for arithmetic operations:

# %%
# Works kinda like the JOIN operations in SQL
obj2 = obj0 + obj1

# %% [markdown]
# Both Series object & its index have a name property:

# %%
obj2.name = "Covid Cases"
obj2.index.name = "State"

# %%
test_obj.index = ["Rick", "Bilbo", "Thorin", "Elrond"]

# %% [markdown]
# ### DataFrame

# %%
data_set = {"state": ["Utah", "California", "Ohio", "California", "Utah", "Ohio"], "year": [2000, 2001, 2002, 2001, 2002, 2003], "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# Columns param rearranges the cols
df = pd.DataFrame(data_set, columns=("year", "state", "pop", "debt"))

# %%
# First 5 records
df.head()

# %%
# Last 5 records
df.tail()

# %%
df.columns

# %% [markdown]
# The column can be accessed in a dict-like notation:

# %%
df["pop"]

# %%
df["debt"] = (np.abs(np.random.standard_normal(len(df))) * 500).round(0).astype(np.int16)

# %%

