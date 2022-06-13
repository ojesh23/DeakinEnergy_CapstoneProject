import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
import datetime
import numpy as np
import calendar
from datetime import date

#these two options will be linked with the drop down list on the webpage
festival = 'Easter'
year = '2017'

#main algo starts from here

#headers for fixing error 403
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

#code.activestate.com
def calc_easter(year):
    "Returns Easter as a date object."
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    
    return date(year, month, day)

#checking input to decide day and month
if(festival == 'New Year'):
    month = '01'
    day=date(year, int(month), 1)
if(festival == 'Christmas'):
    month='12'
    day=date(int(year), int(month), 25)
if(festival == 'Australia Day'):
    month='01'
    day=date(year, int(month), 26)
if(festival == 'Anzac Day'):
    month='04'
    day=date(year, int(month), 25)
if(festival == 'Easter'):
    day=calc_easter(int(year))
    month=str(0)+str(day.month)
if(festival == 'Good Friday'):
    day=calc_easter(int(year))-datetime.timedelta(days=2)
    month=str(0)+str(day.month)

#setting prev and next day from the festival
day1=day-datetime.timedelta(days=1)
day2=day+datetime.timedelta(days=1)

#states array
states = ['VIC1',
          'NSW1',
          'SA1',
          'QLD1',
          'TAS1']

#demands array initially empty
demands = []
demands1 = []
demands2 = []
average = []

#calculating the demands
for state in states:
	#adding month and year to the link
    url = 'https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{}{}_{}.csv'.format(year,month,state)
    s = requests.get(url, headers = headers).text
    df=pd.read_csv(StringIO(s), sep=',')
    
    df[['Date','Time']] = df.SETTLEMENTDATE.str.split(' ',expand=True)
    df['Date'] = pd.to_datetime(df['Date'])
	
	#filtering data based on day
    average.append(df['TOTALDEMAND'].sum()/(df.shape[0]/2));
    
    #total demand on festival
    x=df.query('Date == @day')
    count=x.shape[0]/2
	#averaging for each hour
    demands.append(x['TOTALDEMAND'].sum()/count)
    
    #total demand a day before
    x=df.query('Date == @day1')
    count=x.shape[0]/2
	#averaging for each hour
    demands1.append(x['TOTALDEMAND'].sum()/count)
    
    #total demand a day after
    x=df.query('Date == @day2')
    count=x.shape[0]/2
	#averaging for each hour
    demands2.append(x['TOTALDEMAND'].sum()/count)

#plotting results
plt.figure(figsize = (10,5))

#setting positions for plots
barWidth = 0.2
r1 = np.arange(len(demands))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2] 
r4 = [x + barWidth for x in r3]

#plotting bars
objects = ('Victoria', 'New South Wales', 'South Australia', 'Queensland', 'Tasmania')
plt.bar(r1,demands1,  width=barWidth,  alpha = 0.8, color = 'green', label='Previous Day ('+calendar.day_name[day1.weekday()]+')')
plt.bar(r2,demands,  width=barWidth, alpha = 0.8, color = 'orange', label=festival+' ('+calendar.day_name[day.weekday()]+')')
plt.bar(r3,demands2,  width=barWidth, alpha = 0.8, color = 'blue', label='Next Day ('+calendar.day_name[day2.weekday()]+')')
plt.bar(r4, average,  width=barWidth,  alpha = 0.8, color = 'chocolate', label='Average')

#setting labels and titles
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(demands))], objects)
plt.xlabel('State')
plt.ylabel('Demand Per Hour (MWh)')
plt.title('Energy Demand per Hour on '+festival+' '+year)
plt.grid(axis = 'both', alpha = .2)

#plotting legends
plt.legend()