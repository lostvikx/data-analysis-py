#%% [markdown]
# # Data Loading, Storage, & File Formats

# %% [markdown]
# `pandas` features a number of functions for reading tabular data as a DataFrame object.

# %%
import pandas as pd
import numpy as np

# %%
ex1 = "examples/ex1.csv"

# %% [bash]
# ! cat $ex1

# %%
test_df1 = pd.read_csv(ex1)
test_df1

#%% [markdown]
# Use `header` parameter to set any row as the column labels or `names` to set custom column labels:

# %%
# Creates column index [0:)
df_col_names = ["a", "b", "c", "d", "message"]
test_df2 = pd.read_csv("examples/ex2.csv", header=None, names=df_col_names)
test_df2

# %% [markdown]
# Let's say we want the "message" column as the index

# %%
pd.read_csv("examples/ex2.csv", names=df_col_names, index_col="message")

# %% [markdown]
# Hierarchical Index:

# %%
pd.read_csv("examples/csv_mindex.csv", index_col=["key1", "key2"])

# %% [markdown]
# Some times a table might not have a fixed or use some other delimiter, we can still parse that data:

# %%
pd.read_csv("examples/ex3.txt", sep="\s+")

# %% [markdown]
# The `sep` argument can take a regular expression as well as a string.

# %% [markdown]
# Because there is one fewer column name (first row) than the columns (all other rows), pandas infers the first column as the index.

# %%
pd.read_csv("examples/ex4.csv", skiprows=[0,2,3])

# %% [markdown]
# Handling missing data while reading a file is important. By default, `pandas` parses blank space, or *sentinels* such as NA & NULL.

# %%
# !cat "data/examples/ex5.csv"
pd.read_csv("examples/ex5.csv")

# %%
pd.read_csv("examples/ex5.csv", na_values=["foo"])

# %% [markdown]
# We can add more sentinels by using the `na_values` argument.

# %%
pd.read_csv("examples/ex5.csv", keep_default_na=False)

# %%
df5 = pd.read_csv("examples/ex5.csv", keep_default_na=False, na_values=["NA", "null", "", "NULL"])
df5

# %%
df5.isna()

# %% [markdown]
# We can even define sentinels specific to columns:

# %%
sentinels = {
  "something": ["two"], 
  "message": ["foo"]
}
pd.read_csv("examples/ex5.csv", na_values=sentinels)

# %%

