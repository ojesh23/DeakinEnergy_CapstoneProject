import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

Path = input("Enter cleaned Data File Path with file name: ")
df = pd.read_csv(Path)


def Filter_data(df):
    column_name = input("Please enter the column name for which you need apply filter condition: ")
    if pd.api.types.is_string_dtype(df[column_name]):
        print()
        print("your selected Attribute is string")
        condition = input("Enter the one of the condition code show below for filter, EQUAL TO - 1 or NOT EQUAL TO - 2")
        res = True
        while res:
            if condition != "1":
                if condition != "2":
                    print("entered response", condition, "is not correct")
                    res = True
                    condition = input("please enter 1 for EQUAL TO condition or 2 for NOT EQUAL TO condition: ")
                else:
                    res = False
            else:
                res = False
        value = input("Please enter the condition Value: ")
        if condition == "1":
            new_data = df[df[column_name] == value]

        elif condition == "2":
            new_data = df[df[column_name] != value]

    elif pd.api.types.is_numeric_dtype(df[column_name]):
        print()
        print("your selected Attribute is numerical")
        print("____work in progress____")

    return new_data


def Filter_actio():
    return input("Do you what to filter data (yes or no)?: ")


Filter_action = Filter_actio()
res = True

while res:
    if Filter_action != "yes":
        if Filter_action != "no":
            print("entered response", Filter_action, "is not correct")
            res = True
            Filter_action = input("please enter yes or no: ")
        else:
            res = False
    else:
        res = False

if filter == "yes":
    new_data = Filter_data(df)
else:
    new_data = df


def sample_size():
    return input("is it 1. Single Sample Test or 2. Two Sample Test?, please enter the number:  ")


sample_size = sample_size()
res = True

while res:
    if sample_size != "1":
        if sample_size != "2":
            print("entered response", sample_size, "is not correct")
            res = True
            sample_size = input("please enter 1 for Single Sample Test or 2 Single Sample Test")
        else:
            res = False
    else:
        res = False

if sample_size == "1":
    significance = float(input("Please enter significance value: "))
    sample_Attribute = input("please enter the attribute of hypothesis: ")
    sample_Attribute_success_value = input("please enter the attribute success value of hypothesis: ")
    sample_success = int(len(new_data[new_data[sample_Attribute] == sample_Attribute_success_value]))
    sample_size = int(len(new_data))
    null_hypothesis = float(input("please enter null hypothesis proportion: "))
    alternative_sign = input(
        "if alternative is Ha < Ho use please enter 'smaller', if Ha > Ho use please enter 'larger', if Ha != Ho "
        "enter 'two-sided'")
    # check our sample against Ho for Ha > Ho
    # for Ha < Ho use alternative='smaller'
    # for Ha != Ho use alternative='two-sided'
    stat, p_value = proportions_ztest(count=sample_success, nobs=sample_size, value=null_hypothesis,
                                      alternative=alternative_sign)
    # report
    print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
    if p_value > significance:
        print("Fail to reject the null hypothesis - we have nothing else to say")
    else:
        print("Reject the null hypothesis - suggest the alternative hypothesis is true")

if sample_size == 2:

    # can we assume anything from our sample
    significance = float(input("Please enter significance value: "))
    # note - the samples do not need to be the same size
    size_a = int(input("Number of observation in sample A: "))
    success_a = int(input("Number of successes in sample A: "))
    size_b = int(input("Number of observation in sample B: "))
    success_b = int(input("Number of successes in sample B: "))

    sample_success_a, sample_size_a = (success_a, size_a)
    sample_success_b, sample_size_b = (success_b, size_b)
    # check our sample against Ho for Ha != Ho
    successes = np.array([sample_success_a, sample_success_b])
    samples = np.array([sample_size_a, sample_size_b])
    # note, no need for a Ho value here - it's derived from the other parameters
    alternative_sign = input(
        "if alternative is Ha < Ho use please enter 'smaller', if Ha > Ho use please enter 'larger', if Ha != Ho "
        "enter 'two-sided'")

    stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative=alternative_sign)
    # report
    print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
    if p_value > significance:
        print("Fail to reject the null hypothesis - we have nothing else to say")
    else:
        print("Reject the null hypothesis - suggest the alternative hypothesis is true")
