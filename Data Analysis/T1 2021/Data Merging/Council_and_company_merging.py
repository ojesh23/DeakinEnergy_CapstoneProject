import pandas as pd
import numpy as np
import math

Path = input("Enter Data File Path with file name")
df = pd.read_csv(Path)
df_new = df.drop(["ClearanceRange", "ClearanceSpace", "AdditionalInformation", "CouncilAdditionalInformation", "CouncilClearanceRange", "CouncilClearanceSpace", "CouncilElectricLineContact", "CouncilGenus", "CouncilNonComplianceCode", "CouncilOtherInfrastructurePresent", "CouncilSingleOrMultipleTrees", "CouncilSpanVoltages", "ElectricLineContact", 'NonComplianceCode', "OtherInfrastructurePresent", "Genus", "SingleOrMultipleTrees", "SpanVoltages"], axis =1)

df_new["MaintanedBy"] = ""
df_new["AdditionalInformation"] = ""
df_new["ElectricLineContact"] = ""
df_new["SingleOrMultipleTrees"] = ""
df_new["Genus"] = ""
df_new["NonComplianceCode"] = ""
df_new["OtherInfrastructurePresent"] = ""
df_new["SpanVoltages"] = ""
df_new["ClearanceRange"] = ""
df_new["ClearanceSpace"] = ""
df = pd.read_csv("cleaned_span_inspections.csv")

Attributes = ["ClearanceRange", "AdditionalInformation", "ClearanceSpace", "ElectricLineContact", "Genus", "NonComplianceCode", "OtherInfrastructurePresent", "SingleOrMultipleTrees", "SpanVoltages"]

def merge(typee, i, Attribute):
    if typee == "Both":
        df_new[Attribute][i] = str(df[Attribute][i]) + ", "+ str(df["Council" + Attribute][i])
    if typee == "Company":
        df_new[Attribute][i] = str(df[Attribute][i])
    if typee == "Council":
        df_new[Attribute][i] = str(df[typee + Attribute][i])
    if df_new[Attribute][i] == "" or df[Attribute][i] == "nan":
        df_new[Attribute][i] = np.nan

for i in range(0, len(df)):
    typee = ""
    if (pd.notnull(df["ClearanceRange"][i]) and pd.notnull(df["CouncilClearanceRange"][i])) or (pd.notnull(df["CouncilAdditionalInformation"][i]) and pd.notnull(df["AdditionalInformation"][i])) or (pd.notnull(df["CouncilClearanceSpace"][i]) and pd.notnull(df["ClearanceSpace"][i])) or (pd.notnull(df["CouncilElectricLineContact"][i]) and pd.notnull(df["ElectricLineContact"][i])) or (pd.notnull(df["CouncilGenus"][i]) and pd.notnull(df["Genus"][i])) or (pd.notnull(df["CouncilNonComplianceCode"][i]) and pd.notnull(df["NonComplianceCode"][i])) or (pd.notnull(df["CouncilOtherInfrastructurePresent"][i]) and pd.notnull(df["OtherInfrastructurePresent"][i])) or (pd.notnull(df["CouncilSingleOrMultipleTrees"][i]) and pd.notnull(df["SingleOrMultipleTrees"][i])) or (pd.notnull(df["CouncilSpanVoltages"][i]) and pd.notnull(df["SpanVoltages"][i])):
        df_new["MaintanedBy"][i] = "Both"
        typee = "Both"
    elif pd.notnull(df["ClearanceRange"][i]) or pd.notnull(df["AdditionalInformation"][i]) or pd.notnull(df["ClearanceSpace"][i]) or pd.notnull(df["ElectricLineContact"][i]) or pd.notnull(df["Genus"][i]) or pd.notnull(df["NonComplianceCode"][i]) or  pd.notnull(df["OtherInfrastructurePresent"][i]) or pd.notnull(df["SingleOrMultipleTrees"][i]) or pd.notnull(df["SpanVoltages"][i]):
        df_new["MaintanedBy"][i] = "Company"
        typee = "Company"
    elif pd.notnull(df["CouncilClearanceRange"][i]) or pd.notnull(df["CouncilAdditionalInformation"][i]) or pd.notnull(df["CouncilClearanceSpace"][i]) or pd.notnull(df["CouncilElectricLineContact"][i]) or pd.notnull(df["CouncilGenus"][i]) or pd.notnull(df["CouncilNonComplianceCode"][i]) or  pd.notnull(df["CouncilOtherInfrastructurePresent"][i]) or pd.notnull(df["CouncilSingleOrMultipleTrees"][i]) or pd.notnull(df["CouncilSpanVoltages"][i]):
        df_new["MaintanedBy"][i] = "Council"
        typee = "Council"
    for j in Attributes:
        merge(typee, i, j)

# Removing Duplicates
for j in Attributes:
    data = df_new[j].astype(str)
    for i in range(len(data)):
        if data[i] == 'nan':
            df_new[j][i] = np.nan
        elif data[i] != np.nan:
            Keywords = data[i].split(", ")
            if len(Keywords) == 2:
                if Keywords[0] == Keywords[1]:
                    df_new[j][i] = Keywords[0]

# saving file in excel format
savefilepath = input("Enter path of save")
df_new.to_csv(savefilepath, index=False)