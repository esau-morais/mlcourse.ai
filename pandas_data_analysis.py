import numpy as np
import pandas as pd

pd.set_option("display.max.columns", 100)
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"

data = pd.read_csv(DATA_URL + "adult.data.csv")
data.head()

# 1. How many men and women (sex feature) are represented in this dataset?
data["sex"].value_counts()

# 2. What is the average age (age feature) of women?
data[data["sex"] == "Female"]["age"].mean()

# 3. What is the percentage of German citizens (native-country feature)?
(data["native-country"] == "Germany").sum() / len(data)

# 4-5. What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature) and those who earn less than 50K per year?
ages1 = data[data["salary"] == ">50K"]["age"]
ages2 = data[data["salary"] == "<=50K"]["age"]
print(
    "The average age of the rich: {0} +- {1} years, poor - {2} +- {3} years.".format(
        round(ages1.mean()),
        round(ages1.std(), 1),
        round(ages2.mean()),
        round(ages2.std(), 1),
    )
)

# 6. Is it true that people who earn more than 50K have at least high school education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)
data[data["salary"] == ">50K"]["education"].unique() # No (this is mind-blowing)

# 7. Display age statistics for each race (race feature) and each gender (sex feature). Use groupby() and describe(). Find the maximum age of men of Amer-Indian-Eskimo race.
for (race, sex), sub_df in data.groupby(["race", "sex"]):
    print(f"Race: {race}, sex: {sex}")
    print(sub_df["age"].describe())

# 8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? Consider as married those who have a marital-status starting with Married (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.
# married men
data[(data["sex"] == "Male")
     & (data["marital-status"].str.startswith("Married"))][
    "salary"
].value_counts(normalize=True)

# single men
data[
    (data["sex"] == "Male")
    & ~(data["marital-status"].str.startswith("Married"))
]["salary"].value_counts(normalize=True)

# 9. What is the maximum number of hours a person works per week (hours-per-week feature)? How many people work such a number of hours and what is the percentage of those who earn a lot among them?
max_load = data["hours-per-week"].max()
print(f"Max time - {max_load} hours./week.")

num_workaholics = len(data[data["hours-per-week"] == max_load])
print(f"Total number of such hard workers {num_workaholics}")

rich_share = (
    float(
        data[(data["hours-per-week"] == max_load) & (data["salary"] == ">50K")].shape[0]
    )
    / num_workaholics
)
print(f"Percentage of rich among them {int(100 * rich_share)}%")

# 10. Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) for each country (native-country).
# simple
for (country, salary), sub_df in data.groupby(["native-country", "salary"]):
    print(country, salary, round(sub_df["hours-per-week"].mean(), 2))

# fancy method 
pd.crosstab(
    data["native-country"],
    data["salary"],
    values=data["hours-per-week"],
    aggfunc=np.mean,
).T
