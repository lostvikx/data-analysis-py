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
import re

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

# %% [markdown]
# ### Discretization & Binning
# 
# Continuous data is often discretized or otherwise separated into 'bins' for analysis. Eg: [1.2, 3.2, 2.8] => (1, 4] -> range len: 3
#
# **Note**: Excludes lower and includes upper by default.

# %%
ages = [28, 20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32, 67]
bins = [18,25,35,60,100]

age_cate = pd.cut(ages,bins)
age_cate

# %%
type(age_cate)

# %%
age_cate.codes

# %%
age_cate.categories

# %%
# Frequencies of categorical data in a Series
age_cate.value_counts()
# OR
pd.value_counts(age_cate)

# %% [markdown]
# **Note**: In the string representation of an interval, a parenthesis means that side is open (exclusive), while the square bracket means it's closed (inclusive).
#
# (18, 25] => 18 isn't included, but 25 is.

# %%
# Right side becomes exclusive:
pd.cut(ages,bins,right=False) # [18, 25)...

# %%
age_group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]

age_cat = pd.cut(ages, bins, labels=age_group_names, right=False)
age_cat

# %%
# Returns pd.Index
age_cat.categories

# %%
uni = np.random.uniform(size=100)
# Gives 4 equally ranage cuts:
pd.cut(uni,4,precision=2).value_counts()
# precision=2, limits the precision to 2 decimal places

# %%
pd.qcut(uni,4,precision=2).value_counts()

# %%
# Using qcuts gives 4 equally allocated ranges:
pd.qcut(np.random.standard_normal(1000),4,precision=2).value_counts()

# %% [markdown]
# **Note**: `.qcut` represents quartile cut
#
# Similar to `.cut`, we can pass custom quartiles from 0 to 1:

# %%
pd.qcut(np.random.standard_normal(1000), [0, 0.2, 0.5, 0.7, 1], precision=2).value_counts()

# %% [markdown]
# **Note**: These discretization functions are useful for quartile & group analysis.

# %% [markdown]
# ### Detecting & Filtering Outliers

# %%
df6 = pd.DataFrame(np.random.standard_normal((1000,4)))
df6.head()

# %%
df6.describe()

# %% [markdown]
# Find values in one of the columns exceeding 3 in absolute value:

# %%
col = df6[2]
col[col.abs() > 3]

# %% [markdown]
# Select the entire row that satisfy the above condition:

# %%
df6[(df6.abs() > 3).any(axis=1)]
# The conditional gives us a boolean df, then any returns True if any True found in a row (axis=1), the filter the data.

# %%
# Return rows where all elements, in that row, is greater than 1.1 in abs value:
df6[(df6.abs() > 1.1).all(axis=1)]

# %% [markdown]
# Cap the values to -3 to +3

# %%
df7 = df6.copy()
# np.sign => (-1 if x < 0), (0 if x == 0), (1 if x > 0)
df7[df7.abs() > 3] = np.sign(df7) * 3

# %%
df7.describe()

# %%
np.sign(df7).head()

# %% [markdown]
# ### Permutation & Random Sampling

# %%
df8 = pd.DataFrame(np.arange(5*7).reshape((5,7)))
df8

# %%
# Same as row length
sim_sampler = np.random.permutation(5)
sim_sampler

# %%
df8.take(sim_sampler)
# OR
df8.iloc[sim_sampler]

# %%
col_sampler = np.random.permutation(7)
col_sampler

# %%
df8.take(col_sampler, axis=1)
#OR
df8[col_sampler]

# %% [markdown]
# Select random subset without replacement (no same rows):

# %%
df8.sample(n=3)

# %%
ser2 = pd.Series([1,0,-4,2,7])
ser2

# %% [markdown]
# Random subset with replacement, pass replace=True:

# %%
ser2.sample(n=10, replace=True)

# %% [markdown]
# ### Dummy Variables

# %%
df9 = pd.DataFrame({"key": ["a", "c", "b", "b", "a", "a"], "data": np.random.standard_normal(6)})
df9

# %%
pd.get_dummies(df9["key"])

# %%
dummies = pd.get_dummies(df9["key"], prefix="key")
# Cannot join with a Series
df9[["data"]].join(dummies)

# %%
movies = pd.read_table("datasets/movielens/movies.dat", sep="::", engine="python", header=None, names=["movie_id", "title", "genre"])
# !cat "datasets/movielens/movies.dat"
movies.head()

# %%
mov_dummies = movies["genre"].str.get_dummies(sep="|")
mov_dummies.iloc[:5, :4]

# %%
movs_df = movies.join(mov_dummies.add_prefix("Genre_"))
movs_df.head()

# %%
movs_df.drop("genre",axis=1).head()

# %%
vals = np.random.uniform(size=10)
print(vals)
test_bins = [0,0.2,0.4,0.6,0.8,1]
# pd.cut(vals,test_bins)

# %%
pd.get_dummies(pd.cut(vals,test_bins))

# %%
df10 = pd.DataFrame({"flat": np.random.standard_normal(6).round(2), "sex": ["male", "male", "female", "male", "female", "female"]})
df10

# %%
# df10["sex"].map({"female":0,"male":1})
# OR
pd.get_dummies(df10["sex"],prefix="sex")

# %% [markdown]
# **Note**: Generally, if you have k possible values for a categorical variable, in this case sex can be 2 possible values: male and female; we use k-1 dummy variables to represent it.

# %%
pd.get_dummies(df10["sex"], prefix="sex").iloc[:, 1:]

# %%
df11 = pd.concat([df9, df10], axis=1)
df11

# %%
# If we pass the entire DataFrame, use drop_first to get that k-1 dummy variables
pd.get_dummies(df11, columns=["key", "sex"], drop_first=True)

# %%
# ### Extensions Data Types

# %%
# nan is makes the entire Series dtype: float64 instead of int64
# Mainly because of backward compatibility reasons
pd.Series([1,2,3,np.nan])

# %%
# dtype=pd.Int64Dtype() or "Int64"
s1 = pd.Series([1,2,3,None],dtype="Int64")
s1

# %%
s1.dtype

# %%
s1[s1.notna()]

# %%
s1[3] is pd.NA

# %%
df12 = pd.DataFrame({
  "a": [1,None,3,4],
  "b": ["one", "two", None, "four"],
  "c": [None, False, True, True]
})
df12

# %%
for t,col_name in zip(["Int64", "string", "boolean"], list("abc")):
  df12[col_name] = df12[col_name].astype(t)

df12

# %%
df12.info()

# %% [markdown]
# ### String Manipulation

# %%
a = "a,b,  vik"
b = [s.strip() for s in a.split(",")]
b

# %%
"::".join(b)

# %%
print(a.find(":")) # Returns -1 if not found
print(a.index(",")) # Throws a ValueError

# %%
"vik" in a

# %%
# Counts the occurances of the substring
a.count("z")

# %%
a.replace(",", "||")

# %% [markdown]
# ### RegEx

# %%
re.split(r"\s+", "foo    bar\t baz  \tqux")

# %% [markdown]
# Create a regex object to use the same expression to many strings:

# %%
text = """Bob bob25@proton.me
Vik vik.negi@gmail.com
Robb robb-stark@winter.got
Ryan ryan_reynolds@hollywood.la"""

regex = re.compile(r"[\w\.\-\_]+@\w+\.\w{2,4}")
regex.findall(text)

# %%
print(regex.sub("exposed", text))

# %%
regex_str = r"([\w\.\-\_]+)@(\w+)\.(\w{2,4})"
sep_reg = re.compile(regex_str)

# %%
m = sep_reg.match("vikram.s.negi@proton.me")
m.groups()

# %%
sep_reg.findall(text)

# %%
# Why no zero count? We may never know!
print(sep_reg.sub(r"=> username: \1, domain: \2, suffix: \3", text))

# %%
name_mail_dict = {name: mail_id for name, mail_id in [re.split(r"\s+", t) for t in text.split("\n")]}
name_mail_dict

# %%
name_mail_dict["Wes"] = np.nan
mail_ser = pd.Series(name_mail_dict)
mail_ser

# %%
mail_ser.isna()

# %%
mail_ser.str.contains("gmail")

# %%
matches = mail_ser.str.findall(regex_str).str[0]
matches

# %%
matches.str.get(1)

# %%
mail_df = mail_ser.str.extract(regex_str)
mail_df.columns = pd.Index(["username", "domain", "suffix"])
mail_df.dropna()

# %% [markdown]
# ### Categorical Data

# %%
# The array of distinct values can be called the categories, dictionary, or levels of the data.
fruits = ["apple", "banana", "grapes"]
ser3 = pd.Series(fruits*3)
ser3

# %%
ser3.value_counts()

# %%
ser4 = pd.Series([0,1,2,1,0]*2)
ser3.take(ser4)

# %%
fruits1 = fruits * 2
n_fruits = len(fruits1)
rng = np.random.default_rng(seed=12345)

fruits_df = pd.DataFrame({
  "basket_id": np.arange(n_fruits),
  "fruit": fruits1,
  "count": rng.integers(3,12,size=n_fruits),
  "weight": rng.uniform(0,5,size=n_fruits).round(2)
})
fruits_df

# %%
fruit_cate = fruits_df["fruit"].astype("category").array
fruit_cate

# %%
type(fruit_cate)

# %%
fruit_cate.categories

# %%
fruit_cate.codes

# %%
dict(enumerate(fruit_cate.categories))

# %%
cate1 = pd.Categorical(["doo", "bar", "zap", "doo", "zap"])
cate1

# %%
pd.Categorical.from_codes([2,1,0,1,2],cate1.categories,ordered=True)

# %%
draws = rng.standard_normal(1000)
bins1 = pd.qcut(draws,4,labels=[f"Q{i}" for i in range(1,5)])

# %%
bins1

# %%
bins1.codes[:5]

# %%
bins1 = pd.Series(bins1,name="quartile")
df13 = pd.Series(draws).groupby(bins1).agg(["count","mean","max","min"]).reset_index()
df13

# %%
df13["quartile"]

# %%
n = 10_000_000
labels = pd.Series(["foo","bar","qux","gif"]*(n//4))
lab_cate = labels.astype("category")
lab_cate

# %%
labels.memory_usage(deep=True)

# %%
lab_cate.memory_usage(deep=True)

# %%
# %timeit labels.value_counts()

# %%
# %timeit lab_cate.value_counts()

# %%
cate_s1 = pd.Series(list("abcd")*2).astype("category")
cate_s1

# %%
cate_s1.cat.set_categories(list("abcde")).value_counts()

# %%
cate_s2 = cate_s1[cate_s1.isin(list("ab"))]
cate_s2

# %%
cate_s2.cat.remove_unused_categories()

# %%
cate_s3 = pd.Series(["dev","man","adn","fin"]*2,dtype="category")
cate_s3
# %%
pd.get_dummies(cate_s3,prefix="jd").iloc[:, :-1]
