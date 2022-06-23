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
data_set = {
  "state": [
    "Utah", "California", "Ohio", "California", "Utah", "Ohio"
  ], 
  "year": [2000, 2001, 2002, 2001, 2002, 2003], 
  "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
}

# Columns param rearranges the cols
df = pd.DataFrame(
  data_set, 
  columns=("year", "state", "pop", "debt")
)

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
  print(test_ser2.iloc[[0, 1]])

# %% [markdown]
# When slicing using loc, the last value is inclusive.

# %%
ser1.loc["a": "c"]

# %%
ser1.loc["a": "b"] = 5

# %%
dt = pd.DataFrame(
  np.arange(16).reshape((4,4)), 
  index=["one", "two", "three", "four"], 
  columns=[
    "New York", "San Francisco", "Fargo", "Los Angeles"
  ]
).T

# %%
dt["two"]

# %%
dt.loc["Fargo"]

# %%
dt.sum(axis=1)

# %%
dt[["three", "one"]].sum(axis=1)

# %%
dt.iloc[2:]
# OR
dt["Fargo":]

# %% [markdown]
# **Note**: DataFrame's default `[]` operator:
# * When used for indexing accepts column names
# * While slicing accepts index names or ints

# %%
dt[dt["two"] > 5]

# %%
dt[dt < 3] = np.nan

# %%
dt[dt.isna()] = dt.mean(axis=1).mean()

# %%
dt = dt.astype(np.int64)

# %% [markdown]
# USing `loc`:

# %%
# Returns a Series when only one index:
dt.loc["New York", ["one", "four"]]

# %%
dt.loc[
  ["Fargo", "Los Angeles"], 
  ["one", "three"]
]

# %% [markdown]
# Using `iloc`:

# %%
# Returns a Series when only one index:
dt.iloc[2, [0, 1]]

# %%
dt.iloc[[1, 3], [2, 3, 0]]

# %% [markdown]
# Boolean arrays can be used in `loc`, but not in `iloc`

# %%
dt.loc[dt["two"] > 5]

# %%
try:
  dt.iloc[dt["one"] > 5]
except Exception as e:
  print(e)

# %%
dt.iloc[:, :2][dt["one"] < 5]

# %%
dt.at["Fargo", "three"]

# %%
dt.iat[2, 2]

# %% [markdown]
# **Note**: To avoid ambiguity in code, always use the `loc` & `iloc` operators.

# %% [markdown]
# ### Pitfalls with chaining indexing

# %%
dt.loc[:, "one"] = 1

# %%
dt.iloc[2] = 3

# %%
# Bad Practice:
# dt.loc[dt["three"] < 5]["three"] = 5
# Good Practice:
dt.loc[dt["three"] < 5, "three"] = 5

# %% [markdown]
# Arithmetic Alignment

# %%
s1 = pd.Series(
  np.random.standard_normal(3).round(1)*8,
  index=list("acd")
)
s2 = pd.Series(
  np.random.standard_normal(4).round(1)*8,
  index=list("abcd")
)

# %%
s1+s2

# %% [markdown]
# The internal data alignment introduces missing values in the label locations that don’t overlap.

# %%
df1 = pd.DataFrame(
  (np.random.random((4,3)) * 10).round(1),
  columns=list("abc"),
  index=["Joe", "Mary", "Carl", "Vik"]
)
df2 = pd.DataFrame(
  (np.random.random((3,3)) * 10).round(1),
  columns=list("acd"),
  index=["Joe", "Noah", "Mary"]
)

# %%
df1+df2

# %% [markdown]
# Since column `"b"`` and `"d"`` don't appear in both DataFrame objects, all their results become `nan`. The same holds true for `Carl`, `Noah`, & `Vik`.

# %%
def rand_df(shape, columns, round=1):
  return pd.DataFrame(
    (np.random.random(shape) * 10).round(round),
    columns=columns,
  )

# %%
df_1 = rand_df((3,4), list("abcd"))
df_1

# %%
df_2 = rand_df((4,5), list("abcde"))
df_2

# %%
df_1+df_2

# %%
df_1.add(df_2, fill_value=0)

# %% [markdown]
# The `add` method is one of the several arithmentic operations available in the pandas library.
#
# In simple words, it adds two `DataFrame` objects together, similar to the simple `+` operator. 
#
# The `fill_value` parameter is important: by default is None, optionally takes a float value. Fill the existing missing (NaN) values, and any new element needed for successful DataFrame alignment.

# %%
# fill_value=1, 1 * num == num
df_1.mul(df_2, fill_value=1)

# %%
a = np.arange(12.0).reshape((3, 4))
a[0]

# %%
a - a[0]
# The substraction is performed on all the rows, this is called as broadcasting.

# %%
dt_frame = pd.DataFrame(
  np.arange(12.).reshape((3,4)), columns=list("abcd"),index=cool_states
)
dt_frame

# %%
ser1 = dt_frame.iloc[0]
ser1

# %%
dt_frame - ser1

# %%
dt_frame

# %%
ser2 = dt_frame["a"]
dt_frame.sub(ser2, axis=0)

# %%
ser3 = pd.Series(np.array([4, 2, 3, 5]), index=list("abcd"))
dt_frame.mul(ser3, axis=1)

# %%
df_3 = pd.DataFrame((np.random.standard_normal((4, 3)) * 5).round(2),columns=list("bde"),index=["Utah", "Ohio", "Texas", "Oregon"])
df_3

# %%
df_3.apply(np.std, axis=1)

# %%
df_3.std(axis=1)

# %% [markdown]
# NumPy uses `ddof=0`

# %% [markdown]
# ### Function Application & Mapping
#
# We can use the `apply` method to apply a custom or pre-built funcion. Most common mathematical functions are built-in, so the use of this is rare.

# %%
def minmax_diff(x):
  """
  A custom function

  Args:
    x: Series
  Returns: Difference of max & min element
  """
  return x.max() - x.min()

# %%
df_3.apply(minmax_diff)

# %%
df_3.apply(minmax_diff, axis=1)

# %%
# We can even pass the function as a lambda fn:
df_3.apply(lambda x: x.max() - x.min())

# %%
def both_minmax(x):
  return pd.Series([x.min(), x.max()], index=("min", "max"))

# %%
df_3.apply(both_minmax)

# %%
df_3.apply(both_minmax, axis=1)

# %% [markdown]
# For an element-wise fn, use `applymap` function.

# %%
df_3.applymap(lambda x: f"{x:.1f}")

# %% [markdown]
# `map` is a array/Series method which iterates over the elements & applies the fn

# %%
df_3["e"].map(lambda x: int(x))

# %% [markdown]
# ### Sorting & Ranking

# %%
ser_s = pd.Series(np.random.standard_normal(4)*8, index=("d", "a", "c", "b"))
ser_s

# %%
ser_s.sort_index()

# %%
# Decending order:
ser_s.sort_index(ascending=False)

# %%
ser_s.sort_values()

# %%
df_s = pd.DataFrame(np.random.standard_normal((2,3)) * 8, index=["two", "one"], columns=list("cab"))
df_s

# %%
df_s.sort_index(axis=1, ascending=False)

# %%
df_s.sort_index()

# %%
ser_s["c"] = np.nan
ser_s

# %%
# na_position: {‘first’, ‘last’}, default ‘last’
ser_s.sort_values(na_position="first")

# %%
df_s1 = pd.DataFrame({
  "a": [0,1,5,0],
  "b": [6,-3,0,4]
})
df_s1

# %%
df_s1.sort_values("a")

# %%
df_s1.sort_values(["a","b"])

# %%
ser_r = pd.Series([2, 4, -3, 4, 5, 2])

# %%
ser_r.rank()

# %%
# Same values get a average rank!
# There are multiple different methods of assigning the ranks
ser_r.rank(method="first")

# %%
ser_r.rank(method="dense")

# %%
df_r = pd.DataFrame({
  "a": (np.random.standard_normal(4)*8).round(0),
  "b": (np.random.standard_normal(4)*8).round(0),
  "c": (np.random.standard_normal(4)*8).round(0),
})
df_r

# %%
df_r.rank(axis=1, method="min")
