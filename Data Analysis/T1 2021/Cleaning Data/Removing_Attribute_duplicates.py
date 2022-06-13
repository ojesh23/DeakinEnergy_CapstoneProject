import pandas as pd
import numpy as np

Path = input("Enter merged Data File Path with file name: ")
Cleaned_data = pd.read_csv(Path)

Duplicate_column = input("Please enter the column name for which you need to remove duplicates: ")

data = Cleaned_data[Duplicate_column].astype(str)

Secondary_separator= input("Do we have any Secondary separator (yes or No): ")

if Secondary_separator == "yes":
    Secondary_separator_char = input("What is your Secondary separator : ")
    Primary_separator_char = input("What is your Primary separator : ")
else:
    Primary_separator_char = input("What is your Primary separator : ")

# Code for merging Secondary separator data.
if Secondary_separator == "yes":
    for i in range(len(data)):
        if data[i] == 'nan':
            Cleaned_data[Duplicate_column][i] = np.nan
        elif data[i] != np.nan:
            Keywords = data[i].split(Secondary_separator_char)
            if len(Keywords) == 2:
                Cleaned_data[Duplicate_column][i] = Keywords[0] + Primary_separator_char + Keywords[1]

else:
    data = Cleaned_data[Duplicate_column].astype(str)
    for i in range(len(data)):
        if data[i] == 'nan':
            Cleaned_data[Duplicate_column][i] = np.nan
        elif data[i] != np.nan:
            Keywords = data[i].split(Primary_separator_char)
            if len(Keywords) > 1:
                res = []
                for j in Keywords:
                    if j not in res:
                        res.append(j)
                Cleaned_data[Duplicate_column][i] = Primary_separator_char.join(res)


savefilepath = input("Enter path of save")
Cleaned_data.to_csv(savefilepath, index=False)