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
  columns=[["Ohio", "Ohio", "Texas"],["blue", "red", "blue"]]
)
df1

# %%
df1.index.names = ["key1","key2"]
df1.columns.names = ["state","color"]
df1

# %%
# check the levels of index or columns
print(f"Index levels: {df1.index.nlevels}")
print(f"Column levels: {df1.columns.nlevels}")

# %%
# Subset of the data:
df1["Ohio"].loc["b"]

#%% [markdown]
# A more intuitive way to create MultiIndex:

# %%
pd.MultiIndex.from_arrays(
  [["Ohio","Ohio","Texas"],["blue","red","blue"]],
  names=["state","color"]
)

#%% [markdown]
# Note: Sometimes we may require to swap or change the order of levels in a MultiIndex

# %%
df1.swaplevel(0,1)
# OR if they have names:
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
# OR
df1.groupby(level=1).sum()

# %%
df1.groupby(level="color",axis=1).sum()

# %% [markdown]
# ### Create index using a column in a DataFrame

# %%
df2 = pd.DataFrame({
  "a": np.random.standard_normal(7).round(2),
  "b": np.random.standard_normal(7).round(2),
  "c": ["one", "one", "one", "two", "two", "three", "three"],
  "d": [1,2,3,1,2,2,3]
})
df2

# %%
# Setting a MultiIndex using two cols
df3 = df2.set_index(["c","d"])
df3

# %% [markdown]
# Note: By default the set_index method drops the column values

# %%
df2.set_index(["c","d"],drop=False)

# %%
df3.reset_index()

# %% [markdown]
# ## Combining & Merging Datasets
# 
# Here are few ways to combine data:
#
# - `pd.merge`: Connect rows in DataFrames based on one or more keys, similar to SQL JOIN operation
# - `pd.concate`: Concatenate or "stack" DataFrames along an axis
# - `pd.combine_first`: Splice together overlapping data to fill in missing values in one df with values from other df

# %%
df4 = pd.DataFrame({
  "key": list("abbcaab"),
  "data": np.random.standard_normal(7).round(2)
})
df4

#%%
df5 = pd.DataFrame({
  "key": list("cad"),
  "data": np.random.standard_normal(3).round(2)
})
df5

# %%
pd.merge(left=df4,right=df5,on="key")

# %% [markdown]
# Note: The merge method drops the keys which aren't included in both objects, it defaults to inner join meaning that it is an intersection of both data tables.
#
# This is an example of a many-to-one join; the data in `df4` has multiple rows of `a`, whereas `df5` has only a single value of `a` and `c`.
#
# Also Note: The column `data` got renamed by adding a suffix.

# %%
pd.merge(df4,df5,on="key",how="outer")

# %% [markdown]
# |Option|Behavior|
# |---|---|
# |`how="inner"`|only the key combinations found on both tables|
# |`how="left"`|key combinations found on left table|
# |`how="right"`|key combinations found on right table|
# |`how="outer"`|all the key combinations found on both table|

# %% [markdown]
# Here is an diagram to illustrate how two tables are merged:
#
# ![pd.merge, how parameter](imgs/types-of-join.png)
#
# Here is an example of many-to-many joining of tables:

# %%
df6 = pd.DataFrame({
  "key": list("ababca"),
  "data": pd.Series(np.arange(6),dtype="Int64")
})
df6

# %%
df7 = pd.DataFrame({
  "key": list("aadbb"),
  "data": pd.Series(np.arange(5),dtype="Int64")
})
df7

# %%
pd.merge(df6,df7,on="key")

# %% [markdown]
# Note: The above is the Cartesian product of the matching keys.
#
# Since there are three "a" rows in the left table and two in the right, the result is the product, that is six "a" rows.

# %%
df8 = pd.DataFrame({
  "key1": ["one", "two", "two", "three", "four"],
  "key2": list("abdcd"),
  "data": np.random.standard_normal(5).round(3)
})
df9 = pd.DataFrame({
  "key1": ["one", "three", "two"],
  "key2": list("acd"),
  "data": np.random.standard_normal(3).round(3)
})

pd.merge(df8,df9,on=["key1","key2"],how="outer")

# %% [markdown]
# ### Merging on Index

# %%
left1 = pd.DataFrame({
  "key":list("abaabc"),
  "data": np.random.standard_normal(6).round(3)
})
left1

# %%
right1 = pd.DataFrame({
  "data": np.random.standard_normal(2).round(3)
}, index=["a", "c"])
right1

# %%
pd.merge(left1,right1,left_on="key",right_index=True)

# %%
pd.merge(left1,right1,left_on="key",right_index=True,how="outer")

# %% [markdown]
# Note: If we want to use index of a table as the key, then use left_index or right_index parameter accordingly.
#
# The following is an example of how to merge two tables, one of which has its keys as a MultiIndex:

# %%
left2 = pd.DataFrame({
  "state": ["MH", "MH", "UK", "UK", "MH"],
  "year": [2000,2001,2000,2001,2002],
  "data": np.random.standard_normal(5).round(3)
})
left2

#%%
right_index = pd.MultiIndex.from_arrays([
  ["MH","UK","UK","UK","MH"],
  [2000,2000,2001,2002,2001]
])

# %%
right2 = pd.DataFrame({
  "event1": pd.Series(np.random.standard_normal(5).round(3),index=right_index).sort_index(level=0),
  "event2": pd.Series(np.random.standard_normal(5).round(3),index=right_index).sort_index(level=0)
})
right2

# %%
pd.merge(left2,right2,how="inner",left_on=["state","year"],right_index=True)

# %% [markdown]
# Merging using two indices as keys is also possible:

#%%
def random_numbers(n):
  return (np.random.standard_normal(n) * 8).round(3)

# %%
left3 = pd.DataFrame({
  "NY": random_numbers(4),
  "LA": random_numbers(4)
},index=list("abcd"))
left3

# %%
right3 = pd.DataFrame({
  "WA": random_numbers(3),
  "SF": random_numbers(3)
},index=list("abc"))
right3

# %%
pd.merge(left3,right3,how="outer",right_index=True,left_index=True)

# %% [markdown]
# DataFrame has a `join` method, to simplify merging by index:

# %%
left3.join(right3,how="outer")

# %% [markdown]
# Note: `join` method performs a left join on the key (on=index).

# %%
left1.join(right1,on="key",lsuffix="_x",rsuffix="_y").sort_values("key")

# %%
df10 = pd.DataFrame({
  "SE": random_numbers(2),
  "DT": random_numbers(2)
},index=list("bd"))
df10

# %%
left3.join([right3,df10],how="outer")

# %%
d1 = np.arange(12).reshape(4,3)
np.concatenate([d1,d1],axis=1)

# %%
s1 = pd.Series(np.arange(2),index=list("ab"),dtype="Int64")
s2 = pd.Series(np.arange(3),index=list("bcd"),dtype="Int64")
s3 = pd.Series(np.arange(2),index=list("de"),dtype="Int64")

# %%
pd.concat([s1,s2,s3],axis=0)

# %%
pd.concat([s1,s2,s3],axis=1)

#%% [markdown]
# Hierarchical index on concatenation

# %%
h_ser1 = pd.concat([s1,s2,s3],axis=0,keys=["one","two","three"])
h_ser1

# %%
h_ser1.unstack()

# %% [markdown]
# Note: While concatenating on along `axis="column"`, the keys become the column headers.

# %%
pd.concat([s1,s2,s3],axis=1,keys=["one","two","three"])

# %%
df11 = pd.DataFrame({
  "one": random_numbers(3),
  "two": random_numbers(3)
},index=list("abc"))

df12 = pd.DataFrame({
  "three": random_numbers(2),
  "four": random_numbers(2)
},index=list("ac"))

pd.concat([df11,df12],axis=1,join="outer",keys=["lvl1","lvl2"])

# %% [markdown]
# Note: Here the `keys` arg is used to create a hierarchical column.
#
# We can even pass in a dict in the concat method, the dict keys will be used for keys option.

# %%
pd.concat({
  "lvl1": df11,
  "lvl2": df12
},axis=1)

# %%
df13 = pd.DataFrame(random_numbers((3,4)),columns=list("abcd"))
df14 = pd.DataFrame(random_numbers((2,3)),columns=list("bca"))

pd.concat([df13,df14],axis=0,ignore_index=True)

# %%

