import pandas as pd

Path = input("Enter Cleaned Data File Path with file name: ")
df = pd.read_csv(Path)

merg_on_df = input("Enter Attribute on what to merge on from cleaned data: ")

Path2 = input("Enter Population Data File Path with file name: ")
df2 = pd.read_csv(Path2)

merg_on_df2 = "POA_CODE_2016"

# Converting Data type to str
df[merg_on_df] = df[merg_on_df].astype(str)

# Cleaning Post codes data
for i in range((len(df2))):
    df2[merg_on_df2][i] = df2[merg_on_df2][i].split("A")[1]

# Taking subset of data from the entire data

df2 = df2[[merg_on_df2,"Tot_P_P"]]

# merging the data set

result = pd.merge(df2, df, left_on= merg_on_df2, right_on=merg_on_df)

result.to_csv("merged_population_inspection_data.csv")