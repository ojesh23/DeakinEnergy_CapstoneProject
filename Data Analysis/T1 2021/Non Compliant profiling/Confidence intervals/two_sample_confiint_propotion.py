
import numpy as np
import scipy.stats as stats
from statsmodels.stats import proportion

def Filter_actio():
    return input("Do you want find confidence interval of 1. Categorical data or 2.Numerical data(please enter the "
                 "number)?: ")


def two_proprotions_confint(success_a, size_a, success_b, size_b, significance=0.05):
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    var = prop_a * (1 - prop_a) / size_a + prop_b * (1 - prop_b) / size_b
    se = np.sqrt(var)

    # z critical value
    confidence = 1 - significance
    z = stats.norm(loc=0, scale=1).ppf(confidence + significance / 2)

    # standard formula for the confidence interval
    # point-estimtate +- z * standard-error
    prop_diff = prop_b - prop_a
    confint = prop_diff + np.array([-1, 1]) * z * se
    return confint

Filter_action = Filter_actio()
res = True

while res:
    if Filter_action != "1":
        if Filter_action != "2":
            print("entered response", Filter_action, "is not correct")
            res = True
            Filter_action = input("please enter 1 for Categorical or 2 for Numerical: ")
        else:
            res = False
    else:
        res = False

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

if Filter_action == "1":

    if sample_size == "2":

        size_a = int(input("Number of observation in sample A: "))
        success_a =  int(input("Number of successes in sample A: "))
        size_b =int(input("Number of observation in sample B: "))
        success_b = int(input("Number of successes in sample B: "))
        significance = float(input("Please enter significance value (in proportion, eg=0.05): "))

        confint = two_proprotions_confint(success_a, size_a, success_b, size_b, significance)
        print(confint)

    elif sample_size =="1":
        size_a = int(input("Number of observation in sample: "))
        success_a = int(input("Number of successes in sample: "))
        significance = float(input("Please enter significance value (in proportion, eg=0.05): "))
        ci_low, ci_upp = proportion.proportion_confint(success_a, size_a, alpha=significance, method='normal')
        print(str(round(ci_low * 100, 2)) + "%, " + str(round(ci_upp * 100, 2)) + "%")

elif Filter_action == "2":
    print("Under construction, will be here soon")