#%% [markdown]
# # Data Loading, Storage, & File Formats

# %% [markdown]
# `pandas` features a number of functions for reading tabular data as a DataFrame object.

# %%
import pandas as pd
import numpy as np
import sys
import csv
import json
from lxml import objectify
import requests
import re
import sqlite3
import sqlalchemy as sqla

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
# ! cat "examples/ex5.csv"
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
  # Read chuncksize rows at every iteration
  for chunk in reader:
    # # Check df
    # print(chunk.head())
    # break
    # Adding series (+), fill_value: key_fq was a blank Series.
    key_fq = key_fq.add(chunk["key"].value_counts(), fill_value=0)

key_fq

# %%
with pd.read_csv("examples/ex6.csv", chunksize=1000) as reader:
  # Can iterate like: for chunk in reader.get_chunk()
  # It's a generater function
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

# %% [markdown]
# ### JSON Files

# %%
obj_js = """
{
  "name": "Vikram Negi",
  "cities_lived": ["Navi Mumbai", "Mumbai", "Dehradun"],
  "pet": null,
  "siblings": [
    {"name": "Yogesh", "age": 16, "hobbies": ["gaming", "gym", "anime", "football"]},
    {"name": "Varun", "age": 5, "hobbies": ["reading", "gaming", "cartoon"]}
  ]
}
"""
test_js = json.loads(obj_js)
test_js

# %%
sibs = pd.DataFrame(test_js["siblings"], columns=["name", "age"])
sibs

# %%
# ! cat examples/example.json

# %%
js = pd.read_json("examples/example.json")
js

# %%
js.to_json(sys.stdout)
# Or, for array output:
js.to_json(sys.stdout, orient="records")

# %% [markdown]
# ### HTML & XML: Web Scrapping

# %%
# Need to install lxml module
tables = pd.read_html("examples/fdic_failed_bank_list.html")
failed_banks = tables[0]

# %%
failed_banks.head()

# %% [markdown]
# No. of bank failures by year:

# %%
closing_timestamps = pd.to_datetime(failed_banks["Closing Date"])
closing_timestamps.dt.year.value_counts()

# %%
with open("datasets/mta_perf/Performance_MNR.xml") as f:
  parsed = objectify.parse(f)

# %%
root = parsed.getroot()
root

# %%
xml_data = []
skip_fields = ("INDICATOR_SEQ", "PARENT_SEQ",
               "DESIRED_CHANGE", "DECIMAL_PLACES")

for el in root.INDICATOR:
  el_data = {}
  for child in el.getchildren():
    if child.tag not in skip_fields:
      # using .pyval instead of .text, because it type converts the values
      el_data[child.tag] = child.pyval

  if len(el_data) != 0:
    xml_data.append(el_data)

len(xml_data)

# %%
xml_df = pd.DataFrame(xml_data)
xml_df.head()

# %% [markdown]
# The above data extraction can be done in a single line using `.read_xml` method in pandas:

# %%
xml_df2 = pd.read_xml("datasets/mta_perf/Performance_MNR.xml")

include_fields = []
for field in xml_df2.columns:
  if field not in skip_fields:
    include_fields.append(field)

xml_df2.loc[:, include_fields].head()

# %% [markdown]
# ### Binary Data Format

# %%
# Method 1: Pickle format
df_1 = pd.read_csv("examples/ex1.csv")
df_1

# %%
df_1.to_pickle("examples/ex_pickle")

# %%
pd.read_pickle("examples/ex_pickle")

# %% [markdown]
# `pickle` is only recommended as a short-term storage format, because it may not be supported in the future version of python.

# %% [markdown]
# ### Excel files

# %%
# Used when multiple sheets in an xlsx file
xlsx = pd.ExcelFile("examples/ex1.xlsx")
excel_sheet_name = xlsx.sheet_names[0]
xlsx.parse(excel_sheet_name, index_col=0)

# %%
# OR simply use read_excel
xl_frame = pd.read_excel("examples/ex1.xlsx", sheet_name=excel_sheet_name, index_col=0)

# %% [markdown]
# Similarly there are two ways to write data to an excel file:

# %%
xlsx_writer = pd.ExcelWriter("examples/ex2.xlsx")
xl_frame.to_excel(xlsx_writer, "Sheet1")
xlsx_writer.save()

# %%
pd.read_excel("examples/ex2.xlsx", sheet_name="Sheet1", index_col=0)

# %% [markdown]
# ### Hierarchial Data Format
# 
# First install `tables` package

# %%
test_df = pd.DataFrame(np.random.standard_normal((100,5))*5, columns=list("abcde"))
test_df.head()

# %%
# Store class
store = pd.HDFStore("examples/test.h5")
store

# %%
store["test_1"] = test_df # store DataFrame
store["col_a"] = test_df["a"] # store Series

# %%
store["test_1"].head()

# %% [markdown]
# `HDFStore` supports two storage schemas (format), "`fixed`" & "`table`"
#
# * Default is `fixed`
# * `table` is slower, but supports query operations using special syntax

# %%
# put is the same as assigning store["key_name"] = df, but allows for defining
# specific format
store.put("test_2", test_df, format="table")

# %%
# Query it kinda like a database
store.select("test_2", where=["index >= 10 and index < 15"])

# %% [markdown]
# Here is a simple way to do read & write HDF:

# %%
# Write operation
test_df.to_hdf("examples/test2.h5", "test", format="table")

# %%
# Read operation
pd.read_hdf("examples/test2.h5", "test", where=["index < 5"])

# %% [markdown]
# **Note**: `HDF5` isn't a database. It's best suited for write-once, read-many datasets.

# %% [markdown]
# ### Interacting with Web APIs

# %%
reddit_tiktok_url = "https://www.reddit.com/r/TikTokCringe/top.json?limit=10"
headers = {"user-agent": "Linux Machine (Vikram Singh Negi)"}

res = requests.get(reddit_tiktok_url, headers=headers)
try:
  res.raise_for_status()
  data = res.json()["data"]
except Exception as err:
  print(f"HTTP Error: {err}")

# %%

# pd.DataFrame([child["data"] for child in data["children"]])
[child["data"] for child in data["children"]][0]

# %%
dt = []
valid_fields = ["subreddit", "title", "thumbnail", "url_overridden_by_dest", "subreddit_id", "author", "url", "media", "is_video"]

for child in data["children"]:
  if child["data"]["is_video"]:
    child_data = {}
    for key, val in child["data"].items():
      if key in valid_fields:
        if key == "media":
          fallback_url = val["reddit_video"]["fallback_url"]
          child_data["video_url"] = fallback_url
          child_data["audio_url"] = re.sub(r"[\w+\/]DASH_(\d+)", "/DASH_audio", fallback_url)
        else:
          child_data[key] = val
    dt.append(child_data)

pd.DataFrame(dt)

# %% [markdown]
# ### Interacting with Databases

# %%
con = sqlite3.connect("examples/test.db")

# %%
create_table_query = """
  CREATE TABLE us_cities (
    city VARCHAR(20),
    state VARCHAR(20),
    population REAL,
    rating INTEGER
  )
"""
try:
  con.execute(create_table_query)
  con.commit()
except Exception as err:
  print("DB Error:", err)

# %%
db_data = [
  ("Atlanta", "Georgia", 1.25, 6),
  ("Tallahassee", "Florida", 2.6, 3),
  ("Sacramento", "California", 1.7, 5)
]

con.executemany("INSERT INTO us_cities VALUES (?, ?, ?, ?)", db_data)
con.commit()

# %% [markdown]
# Read from database:

# %%
cur = con.execute("SELECT * FROM us_cities")
rows = cur.fetchall()
rows

# %%
cur.description

# %%
pd.DataFrame(rows, columns=[name[0] for name in cur.description])

# %% [markdown]
# Using SQLAlchemy to make the above process simple:

# %%
db = sqla.create_engine("sqlite:///examples/test.db")
pd.read_sql("SELECT * FROM us_cities", db)
