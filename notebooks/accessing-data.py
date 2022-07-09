#%% [markdown]
# # Data Loading, Storage, & File Formats

# %% [markdown]
# `pandas` features a number of functions for reading tabular data as a DataFrame object.

# %%
import pandas as pd
import numpy as np
import sys
import csv

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
#
# Because there is one fewer column name (first row) than the columns (all other rows), pandas infers the first column as the index.

# %%
pd.read_csv("examples/ex4.csv", skiprows=[0,2,3])

# %% [markdown]
# Handling missing data while reading a file is important. By default, `pandas` parses blank space, or *sentinels* such as NA & NULL.

# %%
# !cat "examples/ex5.csv"
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

# %% [markdown]
# ### Reading text (CSV) files in chunks

# %%
# Read small piece of a large file:
pd.options.display.max_rows = 10

# %%
pd.read_csv("examples/ex6.csv")

# %% [markdown]
# Read in chunks of X rows:

# %%
df6_chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)
key_fq = pd.Series([], dtype="int64") # used later
type(df6_chunker)

# %%
for chunk in df6_chunker:
  print(chunk)
  break

#%% [markdown]
# Or open it as a handler, then iterate over the chunks:

# %%
with pd.read_csv("examples/ex6.csv", chunksize=1000) as reader:
  for chunk in reader:
    # # Check df
    # print(chunk.head())
    # break
    # Adding series (+), fill_value: key_fq was a blank Series.
    key_fq = key_fq.add(chunk["key"].value_counts(), fill_value=0)

key_fq

# %%
with pd.read_csv("examples/ex6.csv", chunksize=1000) as reader:
  print(reader.get_chunk())

#%% [markdown]
# ### Writing out text files

# %%
df5

# %%
df5.to_csv("examples/out.csv")
# ! cat examples/out.csv

# %%
# Other delimiters & represent nan values as "NULL":
df5.to_csv(sys.stdout, sep="|", na_rep="NULL")

# %%
# Can disable index & headers
df5.to_csv(sys.stdout, index=False, header=False) # header = bool or list

# %%
# Only a subset of cols
df5.to_csv(sys.stdout, index=False, columns=["a", "b", "c"])

# %%
# ! cat examples/ex7.csv

# %%
pd.read_csv("examples/ex7.csv")

# %%
with open("examples/ex7.csv") as f:
  for line in csv.reader(f):
    print(line)

# %%
with open("examples/ex7.csv") as f:
  lines = list(csv.reader(f))
  header, values = lines[0], lines[1:]
  data_dict = {h: v for h, v in zip(header, zip(*values))}
  print(data_dict)

# %%
class custom_dialect(csv.Dialect):
  # refer the docs
  delimiter = ";"
  lineterminator = "\n"
  quotechar = '"'
  quoting = csv.QUOTE_MINIMAL

# %% [markdown]
# Now, we can write a csv file with a custom format, dialect:

# %%
with open("examples/custom_dialect_file.csv", "w") as file:
  writer = csv.writer(file, dialect=custom_dialect)
  writer.writerow(("one", "two", "three"))
  writer.writerow(tuple("123"))
  writer.writerow(tuple("456"))

# %%

