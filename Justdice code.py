import pandas as pd
import numpy as np
from matplotlib import pyplot

# Importing the CSV files
adspend_df = pd.read_csv('adspend.csv')
payouts_df = pd.read_csv('payouts.csv')
revenue_df = pd.read_csv('revenue.csv')
installs_df = pd.read_csv('installs.csv')

print(adspend_df.head(10))
print(payouts_df.head(10))
print(revenue_df.head(10))
print(installs_df.head(10))

# Check for missing values in each CSV file
print('Missing values in adspend:')
print(adspend_df.isnull().sum())

print('Missing values in payouts:')
print(payouts_df.isnull().sum())

print('Missing values in revenue:')
print(revenue_df.isnull().sum())

print('Missing values in installs:')
print(installs_df.isnull().sum())

#check for a summary of the columns and datatypes
print(installs_df.info())

"""The event_date data type is an object. This will be changed to datetime and sorted
   from oldest to newest"""

#changed event_date from object to datetime
adspend_df['event_date'] = pd.to_datetime(adspend_df['event_date'], dayfirst=True, format='mixed')
payouts_df['event_date'] = pd.to_datetime(payouts_df['event_date'], dayfirst=True, format='mixed')
revenue_df['event_date'] = pd.to_datetime(revenue_df['event_date'], dayfirst=True, format='mixed')
installs_df['event_date'] = pd.to_datetime(installs_df['event_date'], dayfirst=True, format='mixed')

#sort event_date from oldest to newest
adspend_df = adspend_df.sort_values('event_date') 
payouts_df = payouts_df.sort_values('event_date')
revenue_df = revenue_df.sort_values('event_date')
installs_df = installs_df.sort_values('event_date')

"""First we draw a focus on advertising. Analysis of the adspend data will help paint a clearer 
   picture on the cost of advertising on different networks. In this case we advertise on only 2 networks
   (network id 10 and 60). The adspend table will be collapse by grouping the data into those two networks"""

#Group adspend_df by network_id and sum up the amount spent on ads in each network

adspend_value_group = adspend_df.groupby("network_id")["value_usd"].sum().to_frame()

#Group adspend_df by network_id and count number of clients from each network

adspend_client_group = adspend_df.groupby("network_id")["client_id"].count().to_frame()

print(adspend_value_group)

print (adspend_client_group)


#count number of distinct apps installed based on data from the installs table

n = installs_df.app_id.nunique()
  
print("No.of distinct apps installed :", n)

power_BI_reference = """Of the 51 apps Power BI will now be used to check for the most installed apps (installs > 10,000) 
 and least installed apps (installs < 100). See barchart referencing this findings in the presentation document.
 #NOTE: Threshold values of 100 and 10000 were arbitrarily chosen."""

print(power_BI_reference) 

#group installs_df table by event date and count number of installs in each date

installs_event_group = installs_df.groupby("event_date")["install_id"].count().to_frame()

installs_event_group.reset_index(inplace = True)

pd.set_option("display.max_rows", None)

print(installs_event_group)

#Use the seasonal_decompose method to plot trend and seasonality of installs for the period under consideration
#First convert the event_date column to datetime then set it as the index of the dataframe

from statsmodels.tsa.seasonal import seasonal_decompose

installs_event_group['event_date']= pd.to_datetime(installs_event_group.event_date)

installs_event_group = installs_event_group.set_index('event_date')

analysis = installs_event_group[['install_id']].copy()

#decompose series into trend and seasonality
decompose_result_mult = seasonal_decompose(analysis, model="multiplicative", period=52)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot()
pyplot.show()

def check_stationarity(df):

    #Augmented Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    series= df.values
    check_stationarity.result = adfuller(series)
    print('adfstatistic : %f' %check_stationarity.result[0])
    print('P-value : %f' %check_stationarity.result[1])
    for key, value in check_stationarity.result[4].items():
        print('\t%s: %.3f' % (key, value)) 
        check_stationarity.critical = check_stationarity.result[4]    
  

check_stationarity(installs_event_group)

not_stationary ='''from the ADF test it is observed that the test statistic is greater than the critical 
values and the P-value is much greater than the 0.05 threshold, hence, we fail to 
reject the null hypothesis (non-stationarity). The data is not stationary. '''
   
stationary = '''from the ADF test it is observed that the adf statistic is smaller than the critical 
values and the P-value is less than 0.05 which allows for a rejection of null hypothesis 
(non-stationarity) and acceptance of the alternative hypothesis (stationarity)'''

if check_stationarity.result[1] > 0.05 and check_stationarity.result[0] > check_stationarity.critical['1%']:
    print(not_stationary)
else:
    print(stationary)

Timeseries_observation = """Although the AD-fuller test confirms non-stationarity of the data, according to the 
plot, there seems to be no concrete indication of trend or seasonality. Hence, the non-stationarity captured in the number
of apps installed over the period under consideration could be as a result of other factors like cycles, irregular 
fluctuations, or abrupt changes."""

print(Timeseries_observation)

#Identify the month in the period under consideration that provided the most revenue from installs

#group revenue_df by event date and aggregate the value per date
revenue_event_group = revenue_df.groupby("event_date")["value_usd"].sum().to_frame()

#plot value from installed apps across the period under consideration
revenue_event_group.plot()
pyplot.show()

revenue_event_group.sort_values(by = "value_usd", ascending=False, inplace = True)

top_10_revenue = revenue_event_group.head(10)
print(top_10_revenue)

revenue_statement = """It can be observed that the top 10 dates with the most revenue from installs all fall in November"""

print(revenue_statement)






