import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO

#these two options will be linked with the drop down list on the webpage
month = "12"
state = "VIC1"

#main algo starts from here

#headers for fixing error 403
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

#user options (can be changes)
months = ['January',
          'February',
          'March',
          'April',
          'May',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December']

#empty demands array initially
demands = []

#calculating demand of that month every year
for year in range(2000, 2020):
#adding month and year to the link
    url = 'https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{}{}_{}.csv'.format(year, month, state)
    s = requests.get(url, headers = headers).text
    df = pd.read_csv(StringIO(s), sep = ",")
	
	#adding monthly demand
    demands.append(df['TOTALDEMAND'].sum())

#plotting
fig, ax = plt.subplots()
label = range(2000, 2020)
plt.plot(label,demands, color='blue')

#adding labels and titles
plt.xlabel('Year')
plt.ylabel('Demand per Month (MWh')
plt.title('Energy Demand in ' + months[int(month)-1] + ' for '+state)
plt.xticks(rotation = 45, fontsize = 10, horizontalalignment = 'center', alpha = .7)
plt.xticks(np.arange(2000, 2020, step = 1))

#scatter plot to add points to graph
plt.scatter(label,demands, color = 'orange')
plt.grid(axis = 'both', alpha = .3)