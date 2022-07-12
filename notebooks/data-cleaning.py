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
df4 = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1,1,2,3,3,4,4]})
df4

# %%
# True if record is a duplicate
df4.duplicated()

# %%
df4.drop_duplicates()

# %%
df4["v1"] = np.arange(7)
df4

# %%
df4.drop_duplicates(subset="k1")

# %%
df4.drop_duplicates(subset=["k1"], keep="last")

# %%
meat_df = pd.DataFrame({"food": ["bacon","pulled pork","bacon","pastrami","corned beef","bacon","pastrami","honey ham","nova lox"], "ounces": np.abs((np.random.standard_normal(9)*8).round(1))})
meat_df

# %% [markdown]
# Let's say you want to map each meat with the animal it came from:

# %%
# Method #1:
meat_to_ani = {
  "bacon": "pig",
  "pulled pork": "pig",
  "pastrami": "cow",
  "corned beef": "cow",
  "honey ham": "pig",
  "nova lox": "salmon"
}

meat_df["animal"] = meat_df["food"].map(meat_to_ani)
meat_df

# %%
# Method #2
meat_df["food"].map(lambda food: meat_to_ani[food])

# %%
def food_mapper(food):
  animal_prods = {
    "pig": ["bacon", "pulled pork", "honey ham"],
    "cow": ["pastrami", "corned beef"],
    "salmon": ["nova lox"]
  }

  for ani, foods in animal_prods.items():
    if food in foods:
      return ani

# %%
meat_df["food"].map(food_mapper)

# %%
s1 = pd.Series([12.1, 8, -999, -1000, 0.8, -999])
s1

# %%
# Value -999 might be a sentinel:
s1.replace(-999, np.nan)

# %%
s1.replace([-999,-1000], np.nan)

# %%
s1.replace([-999,-1000], [np.nan, 0.0])

# %%
# Can also provide a dictionary
s1.replace({-999: np.nan, 1000: 0.0})

# %% [markdown]
# ### Renaming Axes

# %%
df5 = pd.DataFrame(
  data=(np.random.standard_normal((3,4))*8).round(2),
  columns=list("abcd"),
  index=["Ohio", "Colorado", "New York"]
)
df5

# %%
df5.index = df5.index.map(lambda city: city[:4].upper().strip())
df5

# %%
# If you don't want to mutate the original:
df5.rename(mapper=lambda city: city.title(), axis=0)

# %%
# Change both index and column together
df5.rename(
  index=lambda i: i.title(),
  columns=lambda c: c.upper()
)

# %%
df5.rename(columns={"a": "foo", "c": "fizz"}, index={"NEW": "GOLD"}).rename(lambda i: i.title(), axis=0)

# %%

