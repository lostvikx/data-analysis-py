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
df["debt"] = (np.abs(np.random.standard_normal(len(df))) * 500).round(0).astype(np.uint64)

# %%
df["is_eastern"] = df["state"] == "Ohio"

# %%
# Delete col:
del df["is_eastern"]
df.columns

# %%
p_data = {"Utah": {2000: 2.5, 2001: 1.1, 2002: 3.2}, "California": {2001: 3.7, 2002: 4.2}}

frame = pd.DataFrame(p_data)

# %%
frame.T  # Transpose

# %%
# DataFrame doesn't have the name attribute.
frame.index.name = "year"
frame.columns.name = "state"

# %% [markdown]
# We can convert a DataFrame to ndarray:

# %%
frame.to_numpy()

#%% [markdown]
# ### Index Object

# %%
index_obj = pd.DataFrame(np.arange(3), index=["a", "b", "c"], columns=["test"])

#%% [markdown]
# Both the index and columns are instances of Index Object

# %%
index_obj.index

# %%
index_obj.columns

# %% [markdown]
# Index objects are immutable and thus can’t be modified by the user:

# %%
try:
  index_obj.index[0] = "z"
except Exception as e:
  print(e)

# %% [markdown]
# ### Reindexing

# %%
re_obj = pd.Series(np.random.standard_normal(3), index=("a", "b", "c"))

# %%
re_obj.reindex(["c", "a", "d"])

# %%
colors = pd.Series(np.random.standard_normal(3), index=(0,2,3))

# %%
colors.reindex(np.arange(6), method="ffill")

# %%
df_count = pd.DataFrame(np.random.standard_normal((3,3)), index=("a", "b", "c"), columns=("Goa", "Kerla", "Delhi"))
df_count = df_count * 50
cool_index = pd.Index(("b", "d", "a", "c"))

df_count.reindex(index=cool_index)
#OR
df_count.reindex(cool_index, axis="index")

# %%
cool_states = pd.Index(["Goa", "Assam", "Kerla"])

df_count.reindex(columns=cool_states)
# OR
# { "index": 0, "columns": 1 }
df_count.reindex(cool_states, axis=1)

# %% [markdown]
# **Note**: In the above examples, we used the `reindex` method to slice & insert either axes. If we only want to slice the `df` then we can use `loc` operator.

# %%
# This makes slicing super easy!
df_count.loc[["a", "b"], ["Goa", "Kerla"]]

# %%
df_count.drop("c")

# %%
df_count.drop(["c", "a"])

# %%
df_count.drop(["Goa", "Delhi"], axis=1)

# %% [markdown]
# ### Indexing, Selection, & Filtering

# %%
# Series indexing works similar to NumPy indexing:
ser1 = pd.Series(np.arange(4), index=["a", "b", "c", "d"])

# %%
ser1["b"] == ser1[1]

# %%
ser1[:-1]

# %%
ser1[["a", "c"]]

# %%
ser1[ser1 > 1]

# %% [markdown]
# **Note**: While we can select data in this manner, the `loc` operator is preferred.

# %%
ser1.loc[["a", "d"]]

# %%
ser1.loc[ser1 > 2]

# %%
test_ser1 = pd.Series(np.arange(3), index=[1, 2, 0])
test_ser2 = pd.Series(np.arange(3), index=["m", "a", "d"])

test_ser1[[0, 2]]

# %%
test_ser2[[0, 2]]

# %%
test_ser1.loc[[0, 1]]

# %%
# The loc operator works exclusively with labels, as opposed to the integers:
try:
  test_ser2.loc[[0, 1]]
except Exception as e:
  print(e)

# %%

