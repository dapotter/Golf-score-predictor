import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
# Turns off unnecessary warning when df.drop(df.columns[11:].tolist(), index=1, inplace=True) is executed in weather processing
pd.options.mode.chained_assignment = None 
import math
import collections
import pickle
from bs4 import BeautifulSoup, NavigableString, Tag
from requests import get
##from selenium import webdriver
##from selenium.webdriver.common.by import By
##from selenium.common.exceptions import TimeoutException
##from selenium.webdriver.support.ui import WebDriverWait
##from selenium.webdriver.support import expected_conditions as EC
import time
import datetime as dt
from datetime import datetime, time, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import style
import csv
import operator as operator # Allows for adding and subtracting lists without converting to numpy arrays
from itertools import accumulate
import scipy.interpolate
import timeit
from unidecode import unidecode
import string
style.use('fast')

''' Scikit Learn Imports '''
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample # For resampling of minority classes

''' Keras imports '''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
''' SciKit imports for deep learning '''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

##year_start = 1982
##year_end = 2018
##num_years = year_end - year_start + 1



'''#########################################################################################################################'''
# Getting Players data:
def GetPlayersData():
    global year_start, year_end
    
    option = webdriver.ChromeOptions()
    option.add_argument('--incognito')

    driver = webdriver.Chrome(executable_path='C:\Windows\chromedriver_win32\chromedriver.exe', chrome_options=option)
    driver.get('https://www.theplayers.com/past-results.html')
    time.sleep(3) # wait as long as it takes for the data to be loaded
    past_results_dropdown = driver.find_element_by_id('pastResultsYearSelector').click()
    print(past_results_dropdown)
    
    end_dates_list = []
    daily_scores_list = []
    winning_scores_list = []
    for year in range(year_end, year_start-1, -1):
        # Select year on webpage
        yr_string = str(year)
        yr_sel = driver.find_element_by_xpath('//option[@value="'+yr_string+'"]').click()
        time.sleep(2) # wait for selected year's table to populate
        # Grab table data:
        # This doesn't work due to lots of empty b tags: dates_element = driver.find_elements_by_tag_name('b')
        # This works: dates_element = driver.find_elements_by_class_name('header-row')
        dates_element = driver.find_element_by_xpath('//span[@class="header-row"]').text
        chars = [char for char in dates_element]
        for char in chars:
            if char.isdigit() == True:
                date_start_index = dates_element.find(char)
                break

        # Populating daily scores:
        daily_scores_element = driver.find_elements_by_xpath('//td[@class="hidden-small"]')
        daily_scores = [x.text for x in daily_scores_element]
        ##print('daily_scores_str: \n', daily_scores)

        
        date_end_index = dates_element.find(yr_string) + 4 # Add 4 to grab year
        date_string = dates_element[date_start_index:date_end_index]
        date = datetime.strptime(date_string, '%m/%d/%Y')
        end_dates_list.append(date)
        
        # Populate all dates:
        for days_subtract in range(0,4):
            if days_subtract == 0:
                day = 'Sunday'
                day_scores = daily_scores[3::5]
            elif days_subtract == 1:
                day = 'Saturday'
                day_scores = daily_scores[2::5]
            elif days_subtract == 2:
                day = 'Friday'
                day_scores = daily_scores[1::5]
            else:
                day = 'Thursday'
                day_scores = daily_scores[0::5]
            day_date = date - timedelta(days = days_subtract)
            daily_scores_list.append([year, day_date, day, day_scores])
        ##print('daily_scores_list: \n', daily_scores_list)

        # Populating winning scores:    
        winning_scores_element = driver.find_elements_by_xpath('//td[not(contains(@class,"hidden-small"))][3]')
        winning_scores = [x.text for x in winning_scores_element]
        winning_score = int(winning_scores[2])
        winning_scores_list.append([year, date, winning_score])
        ##print('winning_scores_list: \n', winning_scores_list)

    # Every day's score for every year:
    df_daily_scores = pd.DataFrame(daily_scores_list, columns = ['Year', 'Date', 'Day', 'Daily Scores'])
    print('df of all daily scores: \n', df_daily_scores)
    df_daily_scores.set_index('Year', inplace=True) # Allows me to select a score for a particular year
    df_daily_scores.to_pickle('players_daily_scores.pickle')

    # Every year's winning scores:
    df_winning_scores = pd.DataFrame(winning_scores_list, columns=['Year', 'Ending', 'Winning Score'])
    df_winning_scores.set_index('Year', inplace=True) # Allows me to select a score for a particular year
    print('df of all winning scores:\n',df_winning_scores)
    df_winning_scores.to_pickle('players_winning_scores.pickle')
    
    return df_winning_scores, df_daily_scores








'''#########################################################################################################################'''
def GetWeatherData(tournament_dates_file_name, station):
    date_list = []; year_list = []
    with open(tournament_dates_file_name, 'r') as file:
        players_date_object = csv.reader(file)
        # print('date_object:\n', date_object)
        date_list = []
        for row in players_date_object:
            row_list = row[0].split(',')
            print(row_list)
            yr = row_list[0]
            yr_list = [row_list[0]] # Extract the year
            for x in row_list[1:]: # Start at 1 to avoid element 0 which is the year
                date = yr + '-' + x # Create the date list: e.g. ['1994-3-24']
                date_list.append(date) # each year and its dates is a list
                year_list.extend(yr_list)
        print('Date list:\n', date_list)
        print('Year list:\n', year_list)
        file.close()

    ########  Running Selenium on Weather Underground to get weather data  ########
    # WARNING: OVERWRITING IMPORTED DATES LIST WITH MY OWN:
    #year_list = ['2013']
    #date_list = ['2013-5-11']
    all_weather_data = []
    for year, date in zip(year_list, date_list):
        print('players_year=', year)
        print('players_date=', date)
        if station == 'KCRG': # KCRG = Craig Municipal Airport near Ponte Vedra Beach 
            url = 'https://www.wunderground.com/history/daily/us/fl/ponte-vedra-beach/KCRG/date/'+date+'?cm_ven=localwx_history'
        elif station == 'KFOF': # KFOF = Westhampton Beach near Shinnecock Hills
            url = 'https://www.wunderground.com/history/daily/us/ny/westhampton-beach/KFOK/date/'+date+'?cm_ven=localwx_history'
        else:
            print('Warning: no station provided')
            break

        #driver.implicitly_wait(60) # Wait 20 seconds before calling page. Also need a way to clear cookies and call page if timeout.

        ## CLOSING AND OPENING THE DRIVER IS TIME CONSUMING. USE BEAUTIFUL SOUP INSTEAD OF SELENIUM.
        while True:
            try:
                option = webdriver.ChromeOptions()
                option.add_argument('--incognito')
                driver = webdriver.Chrome(executable_path='C:\Windows\chromedriver_win32\chromedriver.exe', chrome_options=option)
                #driver.set_page_load_timeout(40)
                driver.get(url)
                print('Got the data for',date)
                break
            except:
                print('Timeout. Trying pageload again.')
                driver.close()
                
        #driver.get('https://www.wunderground.com/history/daily/us/fl/ponte-vedra-beach/KCRG/date/'+players_date+'?cm_ven=localwx_history')
        driver.implicitly_wait(5)
        
        weather_col_elements = driver.find_elements_by_xpath('//button[@class="tablesaw-sortable-btn"]')
        weather_col_names = [x.text for x in weather_col_elements]
        weather_col_names.insert(0,'Date')
        weather_col_names.insert(0,'Year')
        #print(weather_col_names)

        weather_data_elements = driver.find_elements_by_xpath('//table[@id="history-observation-table"]/tbody/tr/td') # Returns rows of data
        weather_data = [x.text for x in weather_data_elements] # Iterates through each row extracting the text, create list
        #print('weather_data before reshaping:\n', weather_data) # Looks like ['2:00 PM', '71 F', '58 F', '63 %', 'NNE', '12 mph',...]

        # Converting list data to 11 columns to align with 11 strings
        # in the weather_col_names list, then converting back to a list
        np_weather_data = np.asarray(weather_data)
        np_weather_data = np_weather_data.reshape(-1,11)

        # rows = np_weather_data.shape[0] # Getting number of rows out of curiosity
        weather_data = np_weather_data.tolist() # Note: The times will be out of order.
        [sublist.insert(0,date) for sublist in weather_data]
        [sublist.insert(0,year) for sublist in weather_data]
        print('weather_data after reshaping:\n', weather_data) # Looks like [['2:00 PM', '71 F', '58 F', '63 %', 'NNE', '12 mph', '0 mph', '30.2 in', '0.0 in', '', ''], ['12:00 AM', '56 F',...]

        all_weather_data.extend(weather_data)

    # Making a df out of two lists
    df = pd.DataFrame(all_weather_data, columns=weather_col_names)
    print('df before removing empty columns:\n',df)
    # Drop empty columns (pd.DataFrame appears to do this
    # automatically, but leaving here just in case:
    for col in df.columns:
        if df[col].empty is True:
            df.drop(columns=col,inplace=True)

    ########  Pickling df and lists:  ########
    # Pickling df_weather:
    df.to_pickle('weather.pickle')

    # Pickling year_list:
    with open('year_list.pkl', 'wb') as yp:
        pickle.dump(year_list, yp)

    # Pickling date_list:
    with open('date_list.pkl', 'wb') as dp:
        pickle.dump(date_list, dp)
    
    return df, year_list, date_list







'''#########################################################################################################################'''
def ProcessWeatherData(df, year_list, date_list):
    # Moving columns:
    df.reset_index(drop=True, inplace=True) # Undoing the 'Time' index set when weather data was gathered. drop=True avoids a new index 0,1,2... being inserted.
    cols = list(df.columns)
    #print('cols =:\n', cols)
    #print('cols[1] =:\n', cols[1])
    cols = [cols[1]]+[cols[2]]+[cols[0]]+cols[3:] # Moving columns from ['Time','Year','Date',...] to ['Year','Date','Time', ...]
    df = df[cols] # Moving the columns
    df_cols_original = list(df.columns)
    print('##########################################################df_cols_original:\n', df_cols_original)
    df.set_index('Year', inplace=True) # Setting here otherwise dropping everything after 10th column doesn't work
    cols = list(df.columns)
    #print('df after moving cols and setting index to year:\n', df)

    # Dropping 11th column onward may not be doing anything depending on how empty columns were handled in GetWeatherData()
    # Should do this by identifying empty columns rather than blindly dropping col 11 onward.
    df.drop(columns=df.columns[11:].tolist(),inplace=True) # Why am I converting to a list?
    ''' Making a wind degree list from the wind direction list provided in the weather data.
        After removing units below, the new wind list is inserted into df. '''
    wind_list = df['Wind'].values.tolist()
    df.drop(columns='Wind', inplace=True)
    wind_conversion_list = [['CALM',np.NaN],['VAR',np.NaN],['N',0],['NNE',22.5],['NE',45],['ENE',67.5],['E',90],
                          ['ESE',112.5],['SE',135],['SSE',157.5],['S',180],
                          ['SSW',202.5],['SW',225],['WSW',247.5],['W',270],
                          ['WNW',292.5],['NW',315],['NNW',337.5]]

    wind_deg_list = [float(card_deg[1]) for card in wind_list for card_deg in wind_conversion_list if card in card_deg]   
    #print('wind_deg_list:\n', wind_deg_list)
    
    print('********************* df after dropping empty columns ************************\n', df)

    #print('List of df columns:\n', df.columns.tolist())
    # This loop starts at 2nd column of df, going data point by data point removing units and
    # converting to integers or floats.
    for i in range(2,len(df.columns)):
        # s_li = df.iloc[:,i].tolist() <- I believe this also works
        # How to handle if it throws a column name error?: df.iloc[:,0].tolist()
        s_series = df.iloc[:,i] # Converting entire column into a series of strings (s_series)
        int_list = []
        for s in s_series: # For each data point in the series...
            for e in s.split(): # ...for each element in string split at spaces (e.g. s = '68 F', e = '68' or 'F')...
                #print('element: ', e)
                char_list = list(e)   # ...automatically splits e in its characters (e.g. e = '68', char_list = ['6','8'] or ['F']). Used in the else loop below
                #print('char list:\n', char_list)
                if e.isdigit(): # Asks if '30' or '30.1' or 'F' are digits. '30.1' and 'F' fail.
                    #print('int element appended to int_list: ', e)
                    int_list.append(int(e))
                    #print('int_list:', int_list) # int_list e.g. [70, 59,...]
                else:
                    for c in char_list: # For c = '3' or '0' or '.' or '1' or 'F', the one with the decimal is made a float.
                        if c == '.':
                            #print('float element appended to int_list: ', e)
                            int_list.append(float(e))

        #print('Temporary integers and floats list:\n', int_list)
        #print('Length of int_list =', int_list)
        df[df.columns[i]] = int_list
        temp_int_series = df[df.columns[i]] # Replace needs a series for inplace=True to work
        temp_int_series.replace(0,np.nan,inplace=True) # df columns don't like to replace values, either use a whole df or a whole series

    print('********************* df after removing units ************************\n', df)

    df.insert(loc=9, column='Wind', value=wind_deg_list)
    df.insert(loc=10, column='Wind Card.', value=wind_list)
    print('df:\n', df)


    # move Time to leftmost column:
    df.reset_index(inplace=True)
    cols = list(df.columns)
    cols = [cols[2]] + cols[0:2] + cols[3:]
    df = df[cols]
    df.set_index('Time', inplace=True)
    #print('df before moving time to leftmost column:\n', df)
    ##df['Temperature'] = df['Temperature'].replace(0,99) # inplace=True doesn't work here
    # Now that time is in number format, convert it to 24 hour time, then sort ascending:
    df.index = pd.to_datetime(df.index).strftime('%H:%M:%S') # Convert to string in H:M:S format
    df.index = pd.to_datetime(df.index, format= '%H:%M:%S').time # Convert to nonstring time object, same format

    df.reset_index(inplace=True) # Bringing time index out of index position.
    df.rename(columns={'index':'Time'},inplace=True) # Renaming the index because the name is lost when converting to datetime object
    #print('df.Time[5] =',df.Time[5])

    # Move year and date columns back to first and second position, respectively:
    cols = list(df.columns)
    cols = [cols[1]]+[cols[2]]+[cols[0]]+cols[3:] # Moving columns from ['Time','Year','Date',...] to ['Year','Date','Time', ...]
    df = df[cols] # Moving the columns
    df.set_index(['Year', 'Date', 'Time'], inplace=True)
    print('df with Year, Date, Time as indices:\n', df)

    total_sub_list = []
    for year, date in zip(year_list, date_list):
        print('----------------------- Date:',date,'---------------------')
        df_sub = df.loc[year, date] # Extracting individual days one at a time from the df data
        df_sub.sort_index(inplace=True)
        print('df_sub.head() index sorted:\n', df_sub.head())
        df_sub_rows, df_sub_cols = df_sub.shape
        #print('df_sub_rows =', df_sub_rows)

        # Return an array of hours:
        hour_list = [df_sub.index[i].hour for i in range(df_sub_rows)]
        minute_list = [df_sub.index[i].minute for i in range(df_sub_rows)]
        print('hour list:\n', hour_list)
        print('minute list:\n', minute_list)
        dec_time_list = []
        for h,m in zip(hour_list, minute_list): # Use hour and minutes list to create new
            dec_time_list.append(h+(m/60))      # decimal time column DecTime. e.g. 2:12 PM = 2.2
        df_sub['DecTime'] = dec_time_list       # Assign list to df_sub column
        #print('df_sub with DecTime', df_sub)

        df_sub.reset_index(inplace=True) # Need to do this because the following loop needs time as a column title

        ############################ Filling in missing hourly values ##########################################
        # Purpose of the the following loop is to make sure that all hours are populated.
        # If the 5th hour of the day is missing (e.g. 3:53, 4:00, 6:00, ...) then it is filled in.
        # Thus, it inserts missing even hourly rows into df_sub by filling Time and DecTime with values, all
        # other columns are populated with NaNs.
        col_list = df_sub.columns.tolist() # Assign list of df_sub columns to col_list
        vars_list = []
        total_vars_list = [] # Holds all weather variables
        for r in range(1,len(df_sub.DecTime[1:])): # Scanning all rows of df_sub
            # Logic: if difference between two time points = 1 hour and minutes don't equal zero:
            # e.g. if: 3:53-2:53 = 1 hr but minutes = 53 != 0:  enter loop to record hour 3 and 0 minutes
            # e.g. if: 2:00-1:00 = 1 hr and minutes = 0 = 0:    don't enter loop because hr 2 exists.
            # e.g. if: 3:53-3:00 = 0 hr and minutes = 53 != 0:  don't enter loop because hr 3 exists.
            # e.g. if: 3:53-3:12 = 0 hr and minutes = 53 != 0:  don't enter loop because hr 3 exists.
            # e.g. if: 4:00-3:53 = 1 hr and minutes = 0:        don't enter loop because hr 4 exists.
            # e.g. if: 5:00-3:53 = 2 hr and minutes = 0:        don't enter loop because hr 5 exists. BUT 4:00 DOESN'T EXIST.

            #print('DecTime[r] - DecTime[r-1] =', (math.floor(df_sub.DecTime[r]) - math.floor(df_sub.DecTime[r-1])) == 1)
            #print('minute_list[r] =', minute_list[r] != 0)
            if math.floor(df_sub.DecTime[r]) - math.floor(df_sub.DecTime[r-1]) == 1 and minute_list[r] != 0:
                #print('Loop initiated due to missing even hour timepoints')
                dec_time_hr = int(math.floor(df_sub.DecTime[r]))
                time_hr = dt.time(dec_time_hr, 0)
                np_nans = np.empty((1,len(df_sub.columns))) # Make an empty numpy array
                np_nans[:] = np.nan # Populate it with NaNs
                #print('np_nans:\n', np_nans)
                vars_list = np_nans.tolist() # vars_list is now entirely NaNs
                vars_list = vars_list[0]
                #print('NaNs only vars_list:\n', vars_list)
                #print('type(vars_list[2]) =', type(vars_list[2]))
                #print('type(vars_list[0,2]) =', type(vars_list[0,2])) # This indexing produces TypeError: list indices must be integers or slices, not tuple
                time_index = col_list.index('Time') # Returns the position or index of 'Time' in col_list
                dec_time_index = col_list.index('DecTime')
                #print('vars_list shape =', np.shape(vars_list))
                # vars_list is a list of lists. c references which list, time_index and dec_time_index reference elements
                vars_list[time_index] = time_hr  # Replaces NaN with hr in vars_list at Time location (first column)
                vars_list[dec_time_index] = dec_time_hr # Replaces NaN with hr in vars_list at DecTime location (last column)
##                print('### non-hourly logic is true at r =',r,'\n')
##                print(' time hr: ', time_hr)
##                print('r =', r)
##                print('col list:\n', col_list)
##                print('time_index =', time_index)
##                print('vars_list:\n', vars_list)
##                print('vars list all nans:\n', vars_list)
##                print('vars_list shape =', np.shape(vars_list))
##                print('dec_time_index =', dec_time_index)

                total_vars_list.append(vars_list)
                #print('total_vars_list:\n', total_vars_list)

        df_t_even = pd.DataFrame(total_vars_list, columns=col_list)
        df_sub = df_sub.append(df_t_even) # All even hours should exist.
##      df.append(dfIns) ## NEVER ITERATIVELY APPEND TO DF. APPEND TO LIST IN LOOP, THEN APPEND TO DF ALL AT ONCE
        df_sub.sort_values(by=['Time'],inplace=True) # With datetime objects added to end, sort the indices
        df_sub.reset_index(drop=True, inplace=True) # The index here is arbitrary integers
        # df_sub's DecTime integer values are all accounted for and sorted, but floats still present for interpolation purposes.
        ####################################################################################################
    
        ########## Now that we have interpolation of NaN values: ###########################################
        #print('*********** df_sub before interpolation: should contain NaNs **************\n', df_sub)
        # Fill in NaNs, interpolate, fill in hourly values, interpolate hourly values, finally delete non hourly:
        np_DecTime = df_sub.DecTime.values
        #df_sub['Wind Gust'] = 0 # Wind Gust and Precip. columns get populated with...
        #df_sub['Precip.'] = 0   # ...all NaNs causing interpolation errors, avoid by filling with zeros.
        for c in ['Temperature','Dew Point','Humidity','Wind Speed','Pressure','Wind','DecTime']: # Column iterator, skips Time, Wind Gust and Precipitation
            np_Col = df_sub[c].values # Need to remove NaNs from here and corresponding indexes from df.DecTime.values
            np_index_NaN = np.array(df_sub.index.get_indexer(df_sub.index[df_sub[c].isnull()])) # Finding NaN indices for each column
            np_XInterp = np.array(df_sub.DecTime[np_index_NaN]) # Define x (DecTime) values to guide the interpolation
            np_Col_noNaN = np.delete(np_Col, np_index_NaN, 0) # Remove NaNs from column of interest
            np_DecTime_noNaN = np.delete(np_DecTime, np_index_NaN, 0) # Remove NaNs from x value column (DecTime)
            np_YInterp = np.interp(np_XInterp, np_DecTime_noNaN, np_Col_noNaN) # Perform interpolation
    ##        print('np_index_NaN:\n',np_index_NaN)
    ##        print('np_XInterp:\n',np_XInterp)
    ##        print('np_DecTime_noNaN:\n',np_DecTime_noNaN)
    ##        print('np_Col_noNaN:\n',np_Col_noNaN)
    ##        df[c][np_index_NaN] = np_YInterp # Chain indexing. Bad.
            df_sub.loc[np_index_NaN,c] = np_YInterp # Insert interpolated values to the correct row and column of df

        # Below: handling Wind Gust and Precipitation data, both of which are too sporadic to interpolate. Moving
        # each data point to its nearest integer DecTime value. Requires finding the indices of DecTime
        # integers.
        np_DecTime = df_sub.DecTime.values
        dectime_val_list = []
        for c in ['Wind Gust','Precip.','Wind Card.']:
            np_Col = df_sub[c].values
            # Need np_index_NaN to create np_Col_noNaN and np_DecTime_noNaN for the column of interest
            np_index_NaN = np.array(df_sub.index.get_indexer(df_sub.index[df_sub[c].isnull()])) # Finding NaN indices for each column
            np_index_noNaN = np.array(df_sub.index.get_indexer(df_sub.index[df_sub[c].notna()])) # array of DecTime float indicies
            #print('np_index_noNaN =', np_index_noNaN)
            np_Col_noNaN = np.delete(np_Col, np_index_NaN, 0)
            np_DecTime_noNaN = np.delete(np_DecTime, np_index_NaN, 0) # Remove NaNs from x value column (DecTime)
            for index_noNaN, dectime in zip(np_index_noNaN, np_DecTime_noNaN): # HERE I'M STEPPING THROUGH NO_NAN VERSIONS OF INDEX AND DECTIME
                ziplist = [index_noNaN, dectime]
                #print('zip list =', ziplist)
                modulo = np.mod(np_DecTime[index_noNaN],1)
                bool_val = np.equal(modulo,0)
                #print('Is dectime value an integer? T/F:', bool_val)
                if bool_val == True:
                    dectime_val_list.extend([dectime]) # Will use later for a list tally of integer dectimes
                if bool_val == False: # If DecTime value isn't an integer
##                    print('Values associated with non-integer DecTime:')
##                    print('dectime =', dectime)
##                    print('index_noNaN =', index_noNaN)
##                    print('np_Col[index_no_NaN] = wind speed or precip =', np_Col[index_noNaN])
##                    print('np_DecTime[index_no_NaN] =', np_DecTime[index_noNaN])
##                    print('#######################################################################\n')
                    if modulo >= 0.5: # If modulo is >= 0.5 (e.g. DecTime is 5.6)
                        rounded_dectime_int = math.ceil(dectime) # Round up the DecTime value
                        dectime_val_list.extend([rounded_dectime_int]) # Keeping a tally of dectimes
                        #print('ceiling rounded dectime int =', rounded_dectime_int)
                        new_index = np.where(np_DecTime==rounded_dectime_int) # Find index of rounded dectime value in np_DecTime
                        #print('new index: row location to insert data point=', new_index[0])
                        np_Col[new_index[0]] = np_Col[index_noNaN] # Write np_Col value to new_index
                        np_Col[index_noNaN] = np.nan # Overwrite old value with NaN
                    else:
                        rounded_dectime_int = math.floor(dectime) # Round dectime down
                        dectime_val_list.extend([rounded_dectime_int]) # Keeping a tally of dectimes
                        #print('floor rounded dectime int =', rounded_dectime_int)
                        new_index = np.where(np_DecTime==rounded_dectime_int) # Find index of rounded dectime value in np_DecTime
                        #print('new index: row location to insert data point=', new_index[0])
                        np_Col[new_index[0]] = np_Col[index_noNaN] # Write np_Col value to new_index
                        np_Col[index_noNaN] = np.nan # Overwrite old value with NaN

            # THIS NEEDS TO BE USED IN THE FUTURE TO AVERAGE WIND GUST AND PRECIP VALUES THAT
            # ALL ROUND TO THE SAME DECTIME VALUE
            # Printing dectime values to see if there are repeat values e.g. multiple 17's or 5 PM's
##            print('dectime_val_list:', dectime_val_list)
##            [print('[index, count]=',[c,dectime_val_list.count(c)]) for c in dectime_val_list]
            df_sub[c] = np_Col # Replace appropriate column with updated values

        #print('********** df_sub after interpolation: should not contain NaNs ************\n', df_sub)
        #########################################################################################################
    
        ########## After Interpolation, remove DecTime's non-whole number float rows: ###########################
        sub_list = [] # Needs to be cleared here before being filled and appended to total_sub_list
        for row in range(len(df_sub.DecTime)):
            if df_sub.DecTime[row].is_integer(): # If DecTime is a whole integer time value...
                df_sub_row_list = df_sub.iloc[row].values.tolist() # ...convert the df's row containing all weather data to a list...
                df_sub_row_list.insert(0,date) # Putting the date...
                df_sub_row_list.insert(0,year) #...and year back into the df
                sub_list.append(df_sub_row_list) # ...and append the list to sub_list

        # Add to the previous date's data that is spaced evenly in time
        total_sub_list.extend(sub_list) # df_total_sub will contain all sub_lists
        
    # Write all date's values spaced evenly in time to a df
    #print('total_sub_list[10:12]:\n', total_sub_list[10:12]) # Don't print entire total_sub_list, it will crash the editor.
    #print('total_sub_list[0][10:12]:\n', total_sub_list[0][10:12])
    # Manually putting the columns in because no previous df has this column lineup to copy from.
    
    cols = ['Year','Date','Time','Temperature','Dew Point','Humidity','Wind Speed','Wind Gust','Pressure','Precip.','Wind','Wind Card.','DecTime']
    df_proc_weather = pd.DataFrame(total_sub_list, columns=cols)

    print('df_proc_weather index values:\n', df_proc_weather[['Date','Time']].to_string())


    # Insert speed and firmness lists:
    n = len(df)
    print('length of df:', n)
    speed_list = [     11.1,11.0,11.3,11.4,11.4,11.6,11.5,11.9,11.8,12.2,12.5, # Thursday 12 AM - 11 AM
                  12.6,12.4,12.5,12.4,12.5,12.6,12.6,12.6,12.4,11.5,10.3,      # Thursday 12 PM to 11 PM
                       10.1,10.2,10.3,10.2,10.4,10.5,10.8,11.1,11.3,11.5,11.3, # Friday 12 AM - 11 AM
                  11.4,11.6,11.5,11.6,11.4,11.5,11.2,11.0, 9.9, 9.6, 9.0, 9.1, # Friday 12 PM - 11 PM
                        9.6, 9.8,10.1,10.2,10.4,10.5,10.7,10.8,10.9,11.3,11.6, # Saturday 12 AM - 11 AM
                  11.7,11.9,12.2,12.4,12.9,13.4,13.9,14.1,14.2,12.1,10.3,      # Saturday 12 PM - 11 PM
                       10.1,10.2,10.2,10.2,10.4,10.3,10.6,10.6,10.8,11.0,10.9, # Sunday 12 AM - 11 AM
                  10.8,10.7,10.7,10.6,10.7,10.5,10.4,10.2, 9.9, 9.8, 9.0     ] # Sunday 12 PM - 11 PM
    
    firm_list = [      0.378,0.379,0.381,0.385,0.382,0.386,0.385,0.388,0.387,0.387,0.388, # Thursday 12 AM - 11 AM
                 0.389,0.393,0.391,0.392,0.394,0.396,0.398,0.397,0.396,0.402,0.370,       # Thursday 12 PM to 11 PM
                       0.366,0.368,0.365,0.367,0.367,0.368,0.369,0.370,0.371,0.370,0.369, # Friday 12 AM - 11 AM
                 0.370,0.369,0.368,0.370,0.371,0.372,0.370,0.369,0.372,0.370,0.370,0.369, # Friday 12 PM - 11 PM
                       0.372,0.374,0.375,0.376,0.378,0.377,0.379,0.380,0.382,0.384,0.387, # Saturday 12 AM - 11 AM
                 0.392,0.401,0.410,0.421,0.434,0.440,0.450,0.457,0.462,0.375,0.370,       # Saturday 12 PM - 11 PM
                       0.363,0.365,0.364,0.367,0.366,0.367,0.365,0.368,0.370,0.369,0.372, # Sunday 12 AM - 11 AM
                 0.371,0.368,0.369,0.367,0.368,0.369,0.370,0.368,0.369,0.365,0.360      ] # Sunday 12 PM - 11 PM

    df_proc_weather.insert(loc=13, column='Speed', value=speed_list)
    df_proc_weather.insert(loc=14, column='Firm', value=firm_list)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df_proc_weather.Time, df_proc_weather.Speed, c='k'); plt.ylabel('Speed, feet')
    plt.subplot(2,1,2)
    plt.plot(df_proc_weather.Time, df_proc_weather.Firm, c='g'); plt.ylabel('Firmness, RU')
    plt.show()

    df_proc_weather.set_index(['Year','Date','Time'], inplace=True)
    
    # Send to pickle:
    df_proc_weather.to_pickle('weather_processed.pickle')
    
    print('***************** Processed weather data after interpolation *****************\n',df_proc_weather)


    
    return df_proc_weather, total_sub_list




'''#########################################################################################################################'''
def ImputeWeatherData(df, year_list, date_list):
    print('df:\n', df)
    df.reset_index(inplace=True)
    df.set_index(['Year','Date'], inplace=True)
    print('Lexsort depth before sorting:', df.index.lexsort_depth)
    # By sorting the index, searching is faster because it allows Pandas
    # to use hash-based indexing. Also avoids a PerformanceWarning.
    df.sort_index(level=1, inplace=True)
    print('Lexsort depth after sorting:', df.index.lexsort_depth)

    ########  Making a minute-to-minute datetime objects list for all tournament days  ########
##    # Not using this at the moment:
##        df_proc_imp_weather.set_index(['Year','Date'], inplace=True)
##        df_proc_imp_weather.index.get_level_values(1).drop_duplicates()
    num_days = len(date_list)
    # date_list contains ['2018-6-14', '2018-6-15', etc]
    # date_list_2 contains ['2018-6-14'X1440, '2018-6-15'X1440, etc]
    # date_list_2, hr_list, min_list should all be 5760 points for a 4 day tournament
    a = date_list
                #[element y for list x in list a for element y in [thing you want to make with x]]
    date_list_subhourly = [y for x in a for y in [x]*1440]
    a = list(range(0,24))*num_days
    hr_list_subhourly = [y for x in a for y in [str(x)]*60] # How to read this? Creates [0x59,1X59,...,23X59]
    a = list(range(0,60))*24*num_days
    minutes_list_subhourly = [y for x in a for y in [str(x)]]
    time_list_subhourly = [y for i,j,k in zip(date_list_subhourly, hr_list_subhourly, minutes_list_subhourly) for y in [i+' '+j+':'+k+':00']]

##    # Keep:
##    print('date_list_2[0:90]', date_list_2[0:90])
##    print('hr_list[0:90]', hr_list[0:90])
##    print('min_list[0:90]', minutes_list[0:90])
##    print('time_list[0:90]', time_list[0:90])
##
##    print('len(date_list_2)', len(date_list_2))
##    print('len(hr_list)', len(hr_list))
##    print('len(min_list)', len(minutes_list))

    ####   BECAUSE datetime_list IS MADE, DELETE CODE FOR YEAR_LIST AND DATE_LIST THAT
    ####   COMPRISES THE FIRST TWO COLUMNS OF df_proc_imp_weather

    datetime_list = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in time_list_subhourly]
    ###########################################################################################



    ########  Initializing lists before for loop  ######## 
    # Master ffits are used for plotting:
    master_ffit_temperature = []; master_ffit_humidity = []; master_ffit_wind_speed = []; master_ffit_wind = []; master_ffit_speed = []; master_ffit_firm = []
    # Fits with noise will go into the final df_proc_imp_weather
    master_ffit_noise = []
    master_ffit_noise_temperature = []; master_ffit_noise_humidity = []; master_ffit_noise_wind_speed = []; master_ffit_noise_wind = []; master_ffit_noise_speed = []; master_ffit_noise_firm = []
    master_year_list = []; master_date_list = []
    raw_temperature = []; raw_humidity = []; raw_wind_speed = []; raw_wind = []; raw_speed = []; raw_firm = []
    time_str_list = []; master_datetime_list = []; master_datetime_subhourly_list = [];
    cumsum_min_list = []; master_cumsum_min_list = []
    master_x_new = []
    ########  For loop iterates through each day imputes Temp, Humidity and Wind Speed  ########
    day_count = 0 # i is used to cumulatively sum the minutes
    for year, date in zip(year_list, date_list):
        current_year = [year]
        current_date = [date]
        time_dt_list = df.loc[year, date].Time.tolist() # Current year-date times (hourly)
        date_str_list = df.loc[year, date].index.values.tolist() # List of tuples of current year-dates (hourly)
        # Overwriting the list I'm iterating through. Shame on me.
        # This new date_str_list contains only dates.
        date_str_list = [i[1] for i in date_str_list]
        #print('time_dt_list:\n', time_dt_list)
        #print('date_str_list:\n', date_str_list)
        
        temperature_list = df.loc[year, date]['Temperature'].tolist()
        humidity_list = df.loc[year, date]['Humidity'].tolist()
        wind_speed_list = df.loc[year, date]['Wind Speed'].tolist()
        wind_list = df.loc[year, date]['Wind'].tolist()
        speed_list = df.loc[year, date]['Speed'].tolist()
        firm_list = df.loc[year, date]['Firm'].tolist()
        #wind_gust_list = df.iloc[year, date]['Wind Gust'].tolist()
        #precip_list = df.iloc[year, date]['Precip.'].tolist()

        min_list = []
        time_str_list = []
        #print('time_dt_list:\n', time_dt_list)
        for t in time_dt_list:
            m = int(dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())/60
            #print('m:\n', m)
            min_list.append(m)
            t_str = t.strftime('%H:%M:%S')  # Converting to string for next loop to build date-time string            
            time_str_list.append(t_str)
        print('min_list:\n', min_list)
        num_minutes = int(min_list[-1] - min_list[0] + 1) # The number of minutes present in the poly fit
        print('Number of minutes in fit =', num_minutes)

        # Summing 0 to first day's min_list, 1440 to second day's min_list, 2880 to third day's min_list, etc.
        # This works even if there are fewer than 1440 minutes in min_list.
        # E.g.  day 2 min_list =        [60, 120, 180,...,1320]
        #       day 2 sum_min_list =    [1440, 1440, 1440,...,1440]
        #       day 2 cumsum_min_list = [1500, 1560, 1620,...,2760]
        # Note that it doesn't finish at 1440*2 = 2880 but that's okay.
        sum_min_list = [day_count*1440]*len(min_list)
        print('sum_list:\n', sum_min_list)
        # Better than using a list comprehension:
        cumsum_min_list = list(map(sum, zip(min_list, sum_min_list)))
        # master_cumsum_min_list is the x-axis for plotting the raw data
        master_cumsum_min_list.extend(cumsum_min_list)
        day_count += 1 # Incrementing for next day
        
        # NOTE: I could put a master_min_list.append(min_list) here but have no use for it later on.
        print('time_str_list:\n', time_str_list)

        # This loop builds a year-list combined string, every element of which is subsequently
        # converted to a Pandas Timestamp object. Not sure if I'm using master_date_time_list
        # for anything below. If not, delete this loop:
        datetime_str_list = []
        for d_str, t_str in zip(date_str_list, time_str_list):
            #print('d_str:', d_str)
            #print('t_str:', t_str)
            d_t_str = d_str + ' ' + t_str
            #print('d_t_str:', d_t_str)
            datetime_str_list.append(d_t_str)
        #print('date_time_str_list:\n', date_time_str_list)
        # Making datetime list from datetime string list:
        datetime_list = pd.to_datetime(datetime_str_list, format='%Y-%m-%d %H:%M:%S')
        datetime_start = datetime_list[0]
        # datetime_subhourly_list has the same time range as x_new below.
        # Eg. Goes from 60,61,62,.. to 1320 but includes the date as well.
        subhourly_min_list = list(range(0, num_minutes))
        ''' Stepping through a list of [0,1,2,...,num_minutes], using each value to construct a
            1-minute timedelta object and adding it to the starting datetime value for that day.
            The result is datetime_subhourly_list, used to build df_proc_imp_weather:
            [...,Timestamp('2018-06-14 01:20:03'), Timestamp('2018-06-14 01:20:04'),
            Timestamp('2018-06-14 01:20:05'), Timestamp('2018-06-14 01:20:06'),...]
            Note: datetime_subhourly_list is the datetime version of x_new: '''
        datetime_subhourly_list = [datetime_start + dt.timedelta(0,x*60,0) for x in subhourly_min_list]
        master_datetime_subhourly_list.extend(datetime_subhourly_list)
##        print('datetime_subhourly_list:\n', datetime_subhourly_list)
##        return datetime_subhourly_list, datetime_subhourly_list
    
        # Minutes of each day: [0, 60, 120, 180, ..., 1380, 0, 60, 120, ...]
        # Because min_list goes back to zero for each day, min_list is not
        # suitable for plotting, need cumulative sum of minutes
        #x = np.asarray(min_list) # X-axis for fitting
        x = np.asarray(cumsum_min_list) # X-axis for fitting

        # Y-axes for fitting
        y_temperature = np.asarray(temperature_list)
        y_humidity = np.asarray(humidity_list)
        y_wind_speed = np.asarray(wind_speed_list)
        y_wind = np.asarray(wind_list)
        y_speed = np.asarray(speed_list)
        y_firm = np.asarray(firm_list)

        # Compiling all raw weather variables for plotting outside of this year-date loop
        raw_temperature.extend(y_temperature)
        raw_humidity.extend(y_humidity)
        raw_wind_speed.extend(y_wind_speed)
        raw_wind.extend(y_wind)
        raw_speed.extend(y_speed)
        raw_firm.extend(y_firm)

        ''' x_new is x-axis with num_minutes number of minutes:
            E.g. if x = [60, 120, 180,...,1320], then
            x_new = [60, 61, 62, 63,...,1320] '''
        #x_new = np.linspace(x[0], x[-1], num=num_minutes)
        x_new = np.linspace(cumsum_min_list[0], cumsum_min_list[-1], num=num_minutes)
        master_x_new.extend(x_new)
    
        '''  Fitting  '''
        ''' Temperature: Polynomial fit '''
        coefs_temperature = poly.polyfit(x, y_temperature, 10)
        ffit_temperature = poly.polyval(x_new, coefs_temperature)
        print('len(ffit_temperature):', len(ffit_temperature))
        master_ffit_temperature.extend(ffit_temperature)
        ''' Humidity: Polynomial fit '''
        coefs_humidity = poly.polyfit(x, y_humidity, 10)
        ffit_humidity = poly.polyval(x_new, coefs_humidity)
        master_ffit_humidity.extend(ffit_humidity)
        ''' Wind Speed: Polynomial fit '''
        coefs_wind_speed = poly.polyfit(x, y_wind_speed, 10)
        ffit_wind_speed = poly.polyval(x_new, coefs_wind_speed)
        master_ffit_wind_speed.extend(ffit_wind_speed)
        ''' Wind Direction: Linear interpolation '''
        func_wind = scipy.interpolate.interp1d(x, y_wind, kind='linear') # Default fit is linear
        ffit_wind = func_wind(x_new)
        master_ffit_wind.extend(ffit_wind)
        ''' Green Speed: Linear interpolation '''
        func_speed = scipy.interpolate.interp1d(x, y_speed, kind='linear') # Default fit is linear
        ffit_speed = func_speed(x_new)
        master_ffit_speed.extend(ffit_speed)
        ''' Green Firmness: Linear interpolation '''
        func_firm = scipy.interpolate.interp1d(x, y_firm, kind='linear') # Default fit is linear
        ffit_firm = func_firm(x_new)
        master_ffit_firm.extend(ffit_firm)

        # TEST: Suppose you have 1000 variables and you want to create poly fits for each one and give
        # each fit function a unique suffix. How can it be done?
        
        ########  Noise Parameters: normalian Distribution  ########
        noise_temperature = np.random.normal(loc=0, scale=0.3, size=num_minutes) # 0.3 F error
        print('len(noise_temperature):', len(noise_temperature))
        noise_humidity =    np.random.normal(loc=0, scale=2.0, size=num_minutes) # 2% error
        noise_wind_speed =  np.random.normal(loc=0, scale=2.2, size=num_minutes) # 1 mph error
        noise_wind =        np.random.normal(loc=0, scale=4.0, size=num_minutes) # 4%*360 deg = 14 deg
        noise_speed =       np.random.normal(loc=0, scale=0.1, size=num_minutes) # 0.1%*12.0 = 0.12 feet
        noise_firm =        np.random.normal(loc=0, scale=0.001, size=num_minutes) # 0.001%*0.370 = 0.00037 RU


        # Adding noise to ffits:
        ffit_noise_temperature =    ffit_temperature + noise_temperature
        ffit_noise_humidity =       ffit_humidity + noise_humidity
        ffit_noise_wind_speed =     ffit_wind_speed + noise_wind_speed
        ffit_noise_wind =           ffit_wind + noise_wind
        ffit_noise_speed =          ffit_speed + noise_speed
        ffit_noise_firm =           ffit_firm + noise_firm


        # Concatentating each day's ffits together:
        master_ffit_noise_temperature.extend(ffit_noise_temperature)
        master_ffit_noise_humidity.extend(ffit_noise_humidity)
        master_ffit_noise_wind_speed.extend(ffit_noise_wind_speed)
        master_ffit_noise_wind.extend(ffit_noise_wind)
        master_ffit_noise_speed.extend(ffit_noise_speed)
        master_ffit_noise_firm.extend(ffit_noise_firm)


    print('master_cumsum_min_list:\n', master_cumsum_min_list)
    # Q: Why does list(zip()) work but not map(list, zip())?
    # Answer: list(zip()) creates a list of tuples. Mapping list onto this
    # list of tuples returns an iterator which is finally converted to a list
    master_ffit_noise = list(map(list,list(zip(master_datetime_subhourly_list, master_ffit_noise_temperature, master_ffit_noise_humidity, master_ffit_noise_wind_speed, master_ffit_noise_wind, master_ffit_noise_speed, master_ffit_noise_firm))))
    #print('master_ffit_noise:\n', master_ffit_noise)
    #print('master_datetime_list:\n', master_datetime_list)
    cols = ['DateTime', 'Temperature', 'Humidity', 'Wind Speed', 'Wind', 'Speed', 'Firm']
    df = pd.DataFrame(master_ffit_noise, columns=cols)
    df.set_index('DateTime', inplace=True)
    df_proc_imp_weather = df
    print('df_proc_imp_weather:\n', df_proc_imp_weather)
    df_proc_imp_weather.to_pickle('weather_processed_imputed.pickle')

    ########  For plotting the raw data over ffits  ######## 
##
##    # Helpful to see Raw Time alongside Cumulative Minutes
##    #raw_time_cumsum_time = [[i, j] for i, j in zip(raw_time, master_cumsum_min_list)]
##    #df_raw_time_cumsum_time = pd.DataFrame(raw_time_cumsum_time, columns=['Time', 'Cumulative Minutes'])
##    #print('df_raw_time_cumsum_time:\n', df_raw_time_cumsum_time)
##
##    # Length of Temp list, Cumulative Minutes list, and Imputed Weather dataframe:
##    #print('len(raw temperature):', len(raw_temperature))
##    #print('len(master cumulative sum min list):', len(master_cumsum_min_list))
##    #print('raw temp and master cumulative sum lengths should be equal')
##    #print('len(df_proc_imp_weather.index):', len(df_proc_imp_weather.index))

##    ########  UNCOMMENT THIS TO RUN  ########
##    ########  Plotting raw data over date-times  ########
##    # This plot results in a lot of whitespace when the data goes from one year to the next.
##    # Not entirely useful. Better is to cumulatively sum minutes and plot the raw hourly weather data
##    # over the ffit and ffit_noise data, which I do below.
##    # If needed, uncomment this section and the plot should work without error.
##    master_date_time_list = [ts.to_pydatetime() for ts in master_date_time_list] # Convert Pandas timestamps to datetime objects
##    # Date-Timestamps to plot raw hourly temperature
##    print('master_date_time_list:\n', master_date_time_list)
##    print('len(master_date_time_list):\n', len(master_date_time_list))
##    # Raw hourly Temperature list copied from dataframe:
##    master_temperature = df.loc[[year_start,year_end, date_start,date_end]]['Temperature'].tolist()
##    plt.figure()
##    plt.scatter(master_date_time_list, master_temperature, s=20, facecolors='none', edgecolors='r', label='Temp')
##    plt.gcf().autofmt_xdate()
##    fmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
##    plt.gca().xaxis.set_major_formatter(fmt)
##    plt.show()

    ########  Plotting ffits, ffits with noise, and raw temperature  ########
    ########  vs minutes over entire time period analyzed  ########
    # If the time period is more than three years of tournaments (e.g. 12 days of play), the data in
    # the plot will be cramped. Zoom into time periods of interest then manually save plot.
    plt.figure()
    
    plt.subplot(1,4,1)
    plt.scatter(master_x_new, master_ffit_temperature, s=5, facecolors='none', edgecolors='k', label='Temp fit')
    plt.scatter(master_x_new, master_ffit_noise_temperature, s=1, color='#ff00ff', alpha=0.5, label='Temp noise fit')
    plt.scatter(master_cumsum_min_list, raw_temperature, s=20, facecolors='none', edgecolors='#00a0a0', label='Temp')
    plt.xlabel('Time, min'); plt.ylabel('Temperature, F')
    plt.ylim(27,90) # Setting min max of y-axis
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390 # 390 is start of play (6:30 AM)
        shade_stop = i*1440 + 1110 # 1110 is end of play (6:30 PM)
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='lower left')

    plt.subplot(1,4,2)
    plt.scatter(master_x_new, master_ffit_humidity, s=5, facecolors='none', edgecolors='k', label='Humidity fit')
    plt.scatter(master_x_new, master_ffit_noise_humidity, s=1, color='#ff00ff', alpha=0.5, label='Humidity noise fit')
    plt.scatter(master_cumsum_min_list, raw_humidity, s=20, facecolors='none', edgecolors='#00a0a0', label='Humidity')
    plt.xlabel('Time, min'); plt.ylabel('Humidity, %')
    plt.ylim(10,130) # Sometimes humidity fit goes over 100
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390
        shade_stop = i*1440 + 1110
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='upper left')

    plt.subplot(1,4,3)
    plt.scatter(master_x_new, master_ffit_wind_speed, s=5, facecolors='none', edgecolors='k', label='Wind Speed fit')
    plt.scatter(master_x_new, master_ffit_noise_wind_speed, s=1, color='#ff00ff', alpha=0.5, label='Wind Speed noise fit')
    plt.scatter(master_cumsum_min_list, raw_wind_speed, s=20, facecolors='none', edgecolors='#00a0a0', label='Wind Speed')
    plt.xlabel('Time, min'); plt.ylabel('Wind Speed, mph')
    plt.ylim(0,40) # Setting min max of y-axis
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390
        shade_stop = i*1440 + 1110
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='upper left')

    plt.subplot(1,4,4)
    plt.scatter(master_x_new, master_ffit_wind, s=5, facecolors='none', edgecolors='k', label='Wind Dir fit')
    plt.scatter(master_x_new, master_ffit_noise_wind, s=1, color='#ff00ff', alpha=0.5, label='Wind Dir noise fit')
    plt.scatter(master_cumsum_min_list, raw_wind, s=20, facecolors='none', edgecolors='#00a0a0', label='Wind Speed')
    plt.xlabel('Time, min'); plt.ylabel('Wind Direction, deg')
    plt.ylim(0,500) # Setting min max of y-axis
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390
        shade_stop = i*1440 + 1110
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='upper left')

    # May be useful additions to the above plotting code:
    plt.subplots_adjust(left=0.08, bottom=0.12, wspace=0.25)

    fig = plt.gcf()
    fig.set_size_inches(15,4) # 10 wide by 3 high
    fig.savefig('Weather Fit with Gaussian Noise - Multiplot - 4 Days.png', bbox_inches='tight')
    plt.show()


    ''' Plotting Speed and Firmness '''
    plt.figure()
    
    plt.subplot(2,1,1)
    plt.scatter(master_x_new, master_ffit_speed, s=5, facecolors='none', edgecolors='k', label='Temp fit')
    plt.scatter(master_x_new, master_ffit_noise_speed, s=1, color='#ff00ff', alpha=0.5, label='Temp noise fit')
    plt.scatter(master_cumsum_min_list, raw_speed, s=20, facecolors='none', edgecolors='#00a0a0', label='Temp')
    plt.xlabel(''); plt.ylabel('Speed, feet')
    plt.ylim(8.8,16.5) # Setting min max of y-axis
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390 # 390 is start of play (6:30 AM)
        shade_stop = i*1440 + 1110 # 1110 is end of play (6:30 PM)
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='upper left')

    plt.subplot(2,1,2)
    plt.scatter(master_x_new, master_ffit_firm, s=5, facecolors='none', edgecolors='k', label='Temp fit')
    plt.scatter(master_x_new, master_ffit_noise_firm, s=1, color='#ff00ff', alpha=0.5, label='Temp noise fit')
    plt.scatter(master_cumsum_min_list, raw_firm, s=20, facecolors='none', edgecolors='#00a0a0', label='Temp')
    plt.xlabel('Time, min'); plt.ylabel('Firmness, RU')
    plt.ylim(0.360,0.500) # Setting min max of y-axis
    plt.xlim(0,1440*4) # (0, 1440*number of days)
    for i in range(0,num_days):
        shade_start = i*1440 + 390 # 390 is start of play (6:30 AM)
        shade_stop = i*1440 + 1110 # 1110 is end of play (6:30 PM)
        plt.axvspan(shade_start, shade_stop, facecolor='gray', alpha=0.1)
    plt.legend(loc='upper left')

    plt.subplots_adjust(left=0.12, bottom=0.12, wspace=0.25)

    fig = plt.gcf()
    fig.set_size_inches(5,5) # 10 wide by 3 high
    fig.savefig('Green Attributes Fit with Gaussian Noise - Multiplot - 4 Days.png', bbox_inches='tight')
    plt.show()
    
    return df_proc_imp_weather # Might also return these: , ffit_noise_temperature, ffit_noise_humidity, ffit_noise_wind_speed, ffit_noise_speed, ffit_noise_firm








'''#########################################################################################################################'''
def MergeTwoWeathers(weather_pickle_str, weather_pickle_str_to_insert):
    df_primary = pd.read_pickle(weather_pickle_str) # Main df into which df_secondary is inserted
    df_secondary = pd.read_pickle(weather_pickle_str_to_insert)
    print('df_primary:\n', df_primary)
    print('df_secondary:\n', df_secondary)

    df_primary.reset_index(drop=True, inplace=True)
    #THIS CODE NEEDED IF DF_PRIMARY AND DF_SECONDARY COLUMNS DON'T ALIGN:
    #cols = list(df_primary.columns)
    #cols = cols[1:3] + [cols[0]] + cols[3:]
    #df_primary = df_primary[cols]
    #print(df_primary.columns)
    #print(df_secondary.columns)
    df_secondary.reset_index(drop=True, inplace=True)

    # Dropping empty columns:
    col_list = [col for col in df_primary.columns if col == '']
    print('df_primary empty columns list=', col_list)
    df_primary.drop(columns=col_list, inplace=True)
    col_list = [col for col in df_secondary.columns if col == '']
    print('df_secondary empty columns list=', col_list)
    df_secondary.drop(columns=col_list, inplace=True)

    # Checking that column orders are the same for both df's:
    print('df_primary columns:\n', df_primary.columns)
    print('df_secondary columns:\n', df_secondary.columns)
    idx = df_primary['Date'].values.tolist().index('2013-5-12') # Finding the row number of the date following the missing date.
    # Notice the indexing on the following line. first section ends at idx-1 inclusive, end section starts at idx inclusive.
    df_merged = pd.concat([df_primary.loc[:(idx-1)], df_secondary, df_primary.loc[(idx):]], ignore_index=True) # FOR SOME REASON NEED TO SUBTRACT 1 FROM
    print('df_merged:\n', df_merged)
    # INDEX BECAUSE IT'S INCLUSIVE. IS THIS UNIQUE TO PANDAS? YES. IT'S BECAUSE IT'S DIFFICULT TO KNOW WHICH ELEMENT COMES NEXT
    # AFTER 'e' OR A DATE.

    print('df_merged:\n', df_merged)
    df_merged.to_pickle('players_weather_merged.pickle')
    return df_merged






'''#########################################################################################################################'''
# Gets tee times from csv files. Will probably move this into GetScorecardsESPN()
def GetTeeTimes(year_list, date_list):
    # Tuples for if statements inside for loop:
    round_tuple = ('Round 1','Round 2','Round 3','Round 4')
    hole_tuple = ('Tee No. 1','Tee No. 10')

    ########  Parsing Tee Times csv  ########
    tee_time_list = []
    df_list = []
    i = 0
    with open('2018_us_open_tee_times.csv', 'r') as file:
        players_tee_time_object = csv.reader(file)
        for row in players_tee_time_object:
            # row: ['2:20 p.m.,Lanto Griffin,Tom Lewis,Jacob Bergeron'] or ['Round 1'] or ['Tee No. 1']
            row_list = row[0].split(',') # Use row[0] because split doesn't operate on lists, requires elements
                
            # Getting day and tee-off hole number from rows in the csv file
            if row_list[0] in round_tuple:
                rnd = row_list[0]
            elif row_list[0] in hole_tuple:
                hole = row_list[0].split()[2]
            else: # No longer dealing with 'Round 4' or 'Tee No. 1' but time strings (e.g. '2:22 PM')
                tee_time = row_list[0]
                tee_time_list.append(tee_time)
                grp_size = int(len(row_list)-1)
                grp_id = int(i)
                i += 1
                for player_name in row_list[1:]: # [1:] avoids tee time element
                    df_list.append([player_name,rnd,hole,grp_size,grp_id,tee_time]) # For building the df
        file.close()

    df = pd.DataFrame(df_list, columns=['Player','Round','Hole','Group Size','Group ID','Tee Time'])
    ########  Making datetime list  ########
    # Currently the times in df are strings. Make a date list containing
    # repeats of the round 1, round 2, round 3, round 4 dates (e.g. '2018-06-14',
    # '2018-6-15', '2018-6-16', '2018-6-17', '2018-6-14', etc)
    # Number of non-duplicate rows is number of players. Recall that
    # some players only have two rounds, so we can't assume everyone has 4 rounds.
    df.set_index(['Player'], inplace=True)
    num_players = int(len(df.index.get_level_values(0).drop_duplicates()))
    df.reset_index(inplace=True)
    # LEARN HOW TO READ THIS CODE: https://stackoverflow.com/questions/49161120/pandas-python-set-value-of-one-column-based-on-value-in-another-column
    # CURRENTLY THIS WON'T WORK FOR MORE THAN A SINGLE YEAR:
    df['Date'] = df['Round']
    # Assigns '2018-06-14 to Date col when Round col contains 'Round 1':
    df.loc[df['Round'] == 'Round 1', 'Date'] = date_list[0]
    df.loc[df['Round'] == 'Round 2', 'Date'] = date_list[1]
    df.loc[df['Round'] == 'Round 3', 'Date'] = date_list[2]
    df.loc[df['Round'] == 'Round 4', 'Date'] = date_list[3]

    date_list_2 = df['Date'].values.tolist()
    tee_time_series = df['Tee Time']
    datetime_list = [y for d,t in zip(date_list_2,tee_time_series) for y in [d+' '+t]]

    #print('datetime_list[0:20] before datetime conversion:\n', datetime_list[0:20])
    datetime_list = [dt.datetime.strptime(i,'%Y-%m-%d %I:%M %p') for i in datetime_list]
    print('datetime_list[0:20] after datetime conversion:\n', datetime_list[0:20])
    print(len(df))
    print(len(datetime_list))
    df.drop(columns=['Tee Time'], inplace=True)
    df.insert(loc=0, column='Tee Time', value=datetime_list)
    #return datetime_list, datetime_list, datetime_list
    ##############################################################

    # Checking the number of rounds played per player. If 1 or 3 rounds
    # has been played then there is likely a spelling error in the player's
    # name or the webpage scrape didn't work:
    counts = df.groupby('Player')['Round'].nunique() # Count the number of unique rounds
    print('counts series:\n', counts)
    # 'counts' is a pandas series, not a dataframe
    # Series containing Player as index and counts of players as a data column:
    # Most players will have 2 or 4 rounds depending on cut status:
    counts_player = counts.index.values.tolist() # The one index is Player
    # The counts series has just one unlabeled column of data with counts,
    # so I don't specify the column title to put into list:
    counts_count = counts.values.tolist() # The one data column of Round counts
    #print('counts_count:\n', counts_count)

    for i,j in zip(counts_count, counts_player):
        if i == 1 or i == 3:
            print('Player has 1 or 3 rounds:', j)

    df.set_index(['Player','Round','Hole','Group Size','Group ID'], inplace=True)
    df.sort_index(level='Player', sort_remaining=True, inplace=True)
    player_list = df.index.get_level_values(0).drop_duplicates().tolist()
    #print('player_list:\n', player_list)

    # Checking df_tee_times for any players with only 2 rounds present,
    # if so append a NaN dataframe to bring total rounds to 4.
    # This is the only place I'm iteratively appending a dataframe and 
    # only because it's easier than pulling the data out of dataframes as lists, 
    # appending to those lists, and reconstructing as a dataframe:
    append_data = [['Round 3',np.NaN,np.NaN,np.NaN,np.NaN],['Round 4',np.NaN,np.NaN,np.NaN,np.NaN]]
    append_cols = ['Round','Hole','Group Size','Group ID','Tee Time']
    df_append = pd.DataFrame(data=append_data, columns=append_cols)
    df_master = pd.DataFrame(columns=['Player','Round','Hole','Group Size','Group ID','Tee Time']) # Empty dataframe
    for player in player_list:
        #print('player:', player)
        df_copy = df.loc[player]
        rounds_played = len(df_copy)
        df_copy.reset_index(inplace=True)
        #print('df_copy:\n', df_copy)
        if rounds_played == 2:
            df_copy = df_copy.append(df_append) # df.append returns a new dataframe, no inplace operation allowed.
            #print('#########  2 rounds played. Appended df_copy:\n', df_copy)
        df_copy.insert(loc=0, column='Player', value=player)
        df_master = df_master.append(df_copy)
    #print('df_master:\n', df_master)
    df = df_master # Returning the dataframe with all rounds filled to df_tee_times
    df.set_index(['Player','Round','Hole','Group Size','Group ID'], inplace=True)

    # Send to pickle:
    df_tee_times = df
    df_tee_times.to_pickle('us_open_tee_times.pickle')

    # Pickling player_list:
    with open('player_list.pkl', 'wb') as plp:
        pickle.dump(player_list, plp)

    print('df tee times:\n', df_tee_times)
    
    return df_tee_times, player_list
    






''' Get player performance stats from an ESPN derived csv file
    http://www.espn.com/golf/statistics/_/year/2018/type/expanded/sort/yardsPerDrive/count/161 '''
def GetPlayerStatsESPN(player_list, player_stats_csv):
    player_stats_list = []
    with open (player_stats_csv, 'r') as tsv:
        player_stats_object = csv.reader(tsv, dialect='excel-tab')
        for i, line in enumerate(player_stats_object):
            #print('line:\n', line) # line looks like: ['Greg Chalmers,45,282.8,58.9,56.9,1.747,50.4']
            if i == 0:
                cols = list(line[0].split(','))
            else:
                row = list(line[0].split(','))
                player_stats_list.append(row)
    tsv.close()

    df = pd.DataFrame(player_stats_list, columns=cols)
    print('df:\n', df)
    tournament_player_list = df.Player.values.tolist()

    ''' player_stats_list: contains players and their stats for 193 players in the PGA
        tournament_player_stats_list: contains players and their stats for only those players in the tournament '''
    tournament_player_stats_list = [p_stat for p_stat in player_stats_list if any(player in p_stat for player in player_list)]
    print('tournament_player_stats_list:\n', tournament_player_stats_list)
    print('len(player_list):\n', len(player_list))
    print('len(tournament_player_stats_list):\n', len(tournament_player_stats_list))

    df = pd.DataFrame(tournament_player_stats_list, columns=cols)
    print('df:\n', df)
    
    # Si Woo Kim is Siwoo Kim on OWGR.com
    # Sung Joon Park is Sunjoon Park on OWGR.com
    # Sung-jae Im is Sungjae Im on OWGR.com
    # Ted Potter Jr. is Ted Potter Jr on OWGR.com
    # WC Liang is Liang Wen-Chong on OWGR.com

    return df


def GetPlayerStatsPGATourWebsite(player_list, pga_player_stats_2018, euro_player_stats_2018):
    df_pga = pd.read_csv('2018_player_stats_pga_csv')
    print('df_pga:\n', df_pga)
    df_euro = pd.read_csv('2018_player_stats_euro_csv')
    print('df_euro:\n', df_euro)

    euro_tee_names_list = df_euro['PlayerTeeSG'].values.tolist()
    euro_app_names_list = df_euro['PlayerAppSG'].values.tolist()
    euro_aro_names_list = df_euro['PlayerAroSG'].values.tolist()
    euro_putt_names_list = df_euro['PlayerPuttSG'].values.tolist()
    euro_drive_dist_names_list = df_euro['PlayerDriveDist'].values.tolist()
    euro_drive_acc_names_list = df_euro['PlayerDriveAcc'].values.tolist()

    new_name_list = []
    for name in euro_tee_names_list:
        if type(name) == str:
            #print('type(name):', type(name))
            #print('name:', name)
            #print('name:', name)
            names = list(name.split(' '))
            #print('names:', names)
            first = names[1]; last = names[0]
            #print('first:', first); print('last:', last)
            first = unidecode(first); last = unidecode(last)
            first = first.lower(); last = last.lower()
            new_name = first + ' ' + last
            new_name = new_name.title()
            new_name_list.append(new_name)
            #print(new_name_list)
        else:
            new_name_list.append(np.NaN)

    df_euro.drop(columns=['PlayerTeeSG'], inplace=True)
    df_euro.insert(loc=0, column='PlayerTeeSG', value=new_name_list)
    print('df_euro:', df_euro)
   
    return df_euro
    




'''#########################################################################################################################'''
# Returns scorecards from ESPN website based on input player list, year list and input tee-off hole list
# This program requires player tee times in order to properly format the scores given that many players
# tee off from the 10th tee in rounds 1 and 2.
def GetScorecardsESPN():
    player_tee_time_list = []
    year = '2018'
    tournament_id = '401025255'
    # Get player id's:
    # Go to tournament leaderboard: http://scores.espn.com/golf/leaderboard?tournamentId=401025255
    # Identify player name link: <a role="button" class="full-name">Brooks Koepka</a>
    # Click on that link
    # Identify Profile and Ranking link: <span>Profile &amp; Ranking</span>
    # Click on it
    # Identify Full Player Profile link: <a href="http://www.espn.com/golf/player/_/id/6798/brooks-koepka" class="button-alt sm">Full Player Profile</a>
    # Pull player id from the link, no need to click on it
    # or.....
    # Go to leaderboard link and pull tbody tags: <tbody id="leaderboard-model-6798">
    
    url = 'http://scores.espn.com/golf/leaderboard?tournamentId='+tournament_id
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    tbody_containers = html_soup.find_all('tbody') # Produces repeating_element_list
    a_containers = html_soup.find_all('a', class_='full-name') # tbody_containers contains player names but this container will be easier to search

    # Checking to see what's in the containers:
    #print('tbody_containers[0:2]:\n', tbody_containers[0:2])
    #print('a_containers[0:2]:\n', a_containers[0:2])

    ''' Pulling player names and id's for the tournament: '''
    player_name_list = []; player_id_list = []; player_name_id_list = []
    for tbody, a in zip(tbody_containers, a_containers): # for repeating_element in repeating_element_list
        tbody_id = tbody.get('id') # Get id element of repeating_element tbody

        # Takes ident_text = 'leaderboard-model-6798', splits it to ['leaderboard', 'model', '6798'],
        # and extract last element '6798'
        player_id = tbody_id.split('-')[-1] # Keeping as strings for forming scorecard URL later on.

        # Takes a tag contents <a class="full-name" role="button">Scott Gregory</a>, extracts text to
        # get 'Scott Gregory':
        player_name = a.text

        player_name_list.append(player_name)
        player_id_list.append(player_id)
        player_name_id_list.append([player_name, player_id])

        # Troubleshooting:
        #print('tbody:\n', tbody)
        #print('ident text:', tbody_id)
        #print('ident number:', ident_num)
        #print('a:\n', a)
        #print('player name:', a_name)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~')

    #print('player_name_id_list:\n', player_name_id_list)
    df_player_name_ids = pd.DataFrame(player_name_id_list, columns=['Player','ID'])

    ''' Building the URL from which score data is scraped: '''
    master_score_list = []; master_hole_list = []; master_yardage_list = []; master_par_list = []
    i_list = list(range(0,len(player_name_list)))
    for i, player_name, player_id in zip(i_list, player_name_list, player_id_list):
        l = len(i_list)
        p = round((i+1)/l*100, 1)
        print('Progress:', p,'%')
        #print('Player name:', player_name)
        #print('Player id:', player_id)

        # Getting player scorecard url:
        url = 'http://www.espn.com/golf/player/scorecards/_/id/'+player_id+'/year/'+year+'/tournamentId/'+tournament_id
        #print('url:\n', url)

        response = get(url)
        #print(response.text[:500])
        
        # Parsing the webpage text and storing as html_soup
        html_soup = BeautifulSoup(response.text, 'html.parser') # Using standard Python html parser
        #print('########################### type(html_soup):\n', type(html_soup))
        #print('########################### html_soup:\n', html_soup)

        active_round_containers = html_soup.find_all(True, class_='roundSwap active')
        #print('########################### active_round_containers:\n', active_round_containers)

    ##    print('########################### score_containers:\n', score_containers)
    ##    print('########################### type(score_containers):\n', type(score_containers))
    ##    print('########################### len(score_containers):\n', len(score_containers))
    ##    first_score = score_containers[0]
    ##    print('########################### first_score:\n', first_score)
    ##    first_score.text
    ##    print('########################### first_score.text:\n', first_score.text)

        # The list 'active_round_containers' holds 1,2,3 or 4 elements, one for each round of golf
        # played (2 if the player gets cut).
        # Each 'row' contains a round's scores buried inside td tags.
        # print('########################## reversed(list(enumerate(active_round_containers))):\n', reversed(list(enumerate(active_round_containers))))
        #print('len(active_round_containers)', len(active_round_containers))

        # Estimating number of rounds played using length of active_round_containers = number of 'roundSwap active' elements.
        num_rounds = len(active_round_containers)
        #print('Number of rounds played:', num_rounds)

##        # Make player_name_index_list equivalent to number of holes played by that player for df indexing
##        player_name_index_list = [player_name]*num_rounds*18 # Repeats the player's name 36 times if 36 holes played
##        player_id_index_list = [player_id]*num_rounds*18 # Repeats the player's id 36 times if 36 holes played

        ''' Getting score data: '''
        # Reshaped to list that is num_holes rows X 4 columns: e.g. 72 holes X 4 rounds or 36 holes X 4 rounds with round 3 and round 4 all NaNs
        reshaped_all_round_score_list = []
        all_round_final_score_list = []
        all_round_score_list = []; all_round_hole_list = []; all_round_yardage_list = []; all_round_par_list = [] # Populated with all holes played in tournament
        final_score_list = [] # Each round's final score
        score_list = []; hole_list = []; yardage_list = []; par_list = [] # Populated with 18 holes of data
        for round_played, row in reversed(list(enumerate(active_round_containers))):
            #print('row length indicates number of rounds played:', len(row))
            #print('row:\n', row)

            ''' This loop pulls each round's final score, which is
                later built into a list independent from each hole's
                score list: '''
            for final_score_element in row.find_all('td', {'rowspan':['2']}):
                final_score_text = final_score_element.text
                final_score = int(final_score_text[0].split(' ')[0])
                print('final_score:', final_score)
                final_score_list.append(final_score)

            ''' This loop pulls each hole's scores and hole numbers
                based on td tags containing specific classes
                known to have each hole's score: '''
            for score in row.find_all('td', {'class':['eagle textcenter','birdie textcenter','par textcenter','bogie textcenter','double textcenter']}):
                #print('td data:\n', data)
                score_num = int(score.text)
                score_list.append(score_num)
            for hole in row.find_all('strong'):
                hole_text = hole.text
                if hole_text.isdigit():
                    hole_num = int(hole_text)
                    hole_list.append(hole_num)

            ''' This loop pulls each hole's yards and par score: '''
            # Not clear on how this loop's code works:
            for br in row.find_all('br'):#.next_sibling:#td', width=True): # Two br tags, one contains hole length, the other is hole par score
                next_s = br.next_sibling
                #print('next_s:\n', next_s)
                if not (next_s and isinstance(next_s, NavigableString)):
                    continue
                next2_s = next_s.next_sibling
                # If next2_s exists and it has a tag and that tag is br
                if next2_s and isinstance(next2_s, Tag) and next2_s.name == 'br':
                    #print('next2_s: ', next2_s)
                    #print('next2_s instance: ', isinstance(next2_s, Tag))
                    #print('next2_s.name: ', next2_s.name)

                    text = str(next_s).strip()
                    if text.isdigit() == True:
                        num = int(text) # num will be either a par number (3,4,5) or a hole yardarge (355) or a 9-hole yardage (3819)
                        if 100 < num < 1000: # Gets hole yardages while avoiding pars and 9-hole yardage numbers in the thousands
                            yardage_list.append(num)
                else:
                    text = str(next_s).strip()
                    if text.isdigit() == True:
                        num = int(text)
                        if num < 6: # If number is a par number (3,4 or 5)
                            par_list.append(num)

            # Verify lengths of all lists are equivalent:
            if len(score_list) != len(hole_list) != len(yardage_list) != len(par_list):
                print('WARNING: Score, Hole, Yardage and Par list lengths are not equivalent.')

            all_round_final_score_list.append(final_score_list)
            all_round_score_list.append(score_list)
            all_round_hole_list.append(hole_list)
            all_round_yardage_list.append(yardage_list)
            all_round_par_list.append(par_list)

            # Clear all lists so they can be populated with the scores of the next round.
            final_score_list = []; score_list=[]; hole_list=[]; yardage_list=[]; par_list=[]

        # Printing all master lists:
        print('all_round_final_score_list:\n', all_round_final_score_list)
##        print('all_round_score_list:\n', all_round_score_list)
##        print('all_round_hole_list:\n', all_round_hole_list)
##        print('all_round_yardage_list:\n', all_round_yardage_list)
##        print('all_round_par_list:\n', all_round_par_list)

        # Populate master lists (not master_score_list yet due to the need to reshape it for the dataframe):
        master_hole_list.extend(all_round_hole_list)
        master_yardage_list.extend(all_round_yardage_list)
        master_par_list.extend(master_par_list)
        
        # Reshape the all round score lists from 18 columns to 2 columns or 4 columns depending on number of rounds played,
        # and putting in player names for indexing:
        reshaped_all_round_score_list = []
        # This code is clunky:
        if num_rounds == 0:
            print('Warning: ',player_name,'\'s profile does not exist')
            np_score0_nans = np.empty((1,18)); np_score0_nans[:] = np.nan; score0_nans = np_score0_nans.tolist()[0] 
            np_score1_nans = np.empty((1,18)); np_score1_nans[:] = np.nan; score1_nans = np_score1_nans.tolist()[0] 
            np_score2_nans = np.empty((1,18)); np_score2_nans[:] = np.nan; score2_nans = np_score2_nans.tolist()[0] 
            np_score3_nans = np.empty((1,18)); np_score3_nans[:] = np.nan; score3_nans = np_score3_nans.tolist()[0] # Make nan list
            all_round_score_list.append(score0_nans)
            all_round_score_list.append(score1_nans)
            all_round_score_list.append(score2_nans)
            all_round_score_list.append(score3_nans)
            
        if num_rounds == 2: # If number of rounds is 2, then third and fourth rounds' nan score lists are populated here
            # Make empty np array; set all elements to nans; convert to list [[nan,nan,nana]] then select inner element [0] = [nan, nan, nana].
            np_score2_nans = np.empty((1,18)); np_score2_nans[:] = np.nan; score2_nans = np_score2_nans.tolist()[0] 
            #print('score2_nans list:\n', score2_nans)
            all_round_score_list.append(score2_nans)
            # Make empty np array; set all elements to nans; convert to list [[nan,nan,nana]] then select inner element [0] = [nan, nan, nana].
            np_score3_nans = np.empty((1,18)); np_score3_nans[:] = np.nan; score3_nans = np_score3_nans.tolist()[0] # Make nan list
            all_round_score_list.append(score3_nans)

        # With all four rounds accounted for, reshape the lists:
        #print('Reshaping all_round_score_list for',num_rounds,'rounds:\n')
        # s0 = score for round 1, s1 is score for round2
        for s0, s1, s2, s3 in zip(all_round_score_list[0], all_round_score_list[1], all_round_score_list[2], all_round_score_list[3]):
            reshaped_all_round_score_list.append([player_name, s0, s1, s2, s3])

        # The reshaped_all_round_score_list is only used to build the master_score_list for all players for the df
        #print('reshaped_all_round_score_list:\n', reshaped_all_round_score_list)
        master_score_list.extend(reshaped_all_round_score_list)
        #print('master_score_list:\n', master_score_list)
    
    df_scorecards = pd.DataFrame(master_score_list, columns=['Player','Round 1','Round 2','Round 3','Round 4'])
    print('df:\n', df_scorecards)

    # Send to pickle:
    df_scorecards.to_pickle('us_open_scorecards.pickle')

    
    return df_scorecards, df_player_name_ids, master_score_list, master_hole_list, master_yardage_list, master_par_list
    


def ScorecardsEDA(df_training_set):
    df_training_set['Score'] = df_training_set['Score'].astype(int)
    df_training_set_hole_groups = df_training_set.groupby('Hole').describe()['Score']
    print('df_training_set_hole_groups description:\n', df_training_set.groupby('Hole').describe()) # Showing each hole's score stats
    print('df_training_set_hole_groups:\n', df_training_set_hole_groups)

    hole_score_group = df_training_set.groupby('Hole')['Score']
    hole_score_list = hole_score_group.index.values.tolist()
    hole_list = hole_score_group.values.tolist()
    print('hole_score_list:\n', hole_score_list)
    print('hole_list:\n', hole_list)
    
    plt.figure()
    plt.suptitle('Hole Avg Scores - All Rounds')
    #ax = fig.add_subplot(111)
    np_score_mean = np.asarray(df_training_set_hole_groups['mean'])
    print('np_score_mean:\n', np_score_mean)
    score_mean_list = df_training_set_hole_groups['mean'].values.tolist()
    print('score_mean_list:\n', score_mean_list)
    x_labels = [str(x) for x in df_training_set_hole_groups.index.get_values().tolist()]
    np_x_labels = np.asarray(x_labels)
    #plt.boxplot(np_score_mean)
    #plt.boxplot(score_mean_list)
    plt.scatter(x_labels, score_mean_list, s=20, facecolors='none', edgecolors='k'); plt.xlabel('Hole'); plt.ylabel('Avg Score')

    #ax.set_xticklabels(np_x_labels)
    #plt.set_xticklabels(np_x_labels)
    plt.show()

    return df_training_set




'''################################################################################################'''
def GetOWGR(player_list, owgr_csv_new, owgr_csv_old):
    world_player_owgr_new_list = []
    with open (owgr_csv_new, 'r') as tsv:
        owgr_object = csv.reader(tsv, dialect='excel-tab')
        for i, line in enumerate(owgr_object):
            #print('line:', line)
            #print('line[0]:', line[0])
            if i == 0:
                cols = list(line[0].split(','))
                #print('cols:', cols)
            else:
                player_owgr_list = list(line[0].split(','))
                player_owgr_list[1] = int(player_owgr_list[1]) # Converting '3' to 3
                world_player_owgr_new_list.append(player_owgr_list)   
    tsv.close()

    world_player_owgr_old_list = []
    with open (owgr_csv_old, 'r') as tsv:
        owgr_object = csv.reader(tsv, dialect='excel-tab')
        for i, line in enumerate(owgr_object):
            #print('line:', line)
            #print('line[0]:', line[0])
            if i == 0:
                cols = list(line[0].split(','))
                #print('cols:', cols)
            else:
                player_owgr_list = list(line[0].split(','))
                player_owgr_list[1] = int(player_owgr_list[1]) # Converting '3' to 3
                world_player_owgr_old_list.append(player_owgr_list)  
    tsv.close()

    ''' Note: player_list_owgr = [['Dustin Johnson', 1], ['Justin Thomas', 2] ,..., ['Aaron Baddeley', 253]]
        Getting the list of tournament players whose names are in the world_player_owgr_list: '''
    tournament_player_owgr_new_list = [p_owgr for p_owgr in world_player_owgr_new_list if any(player in p_owgr for player in player_list)]
    tournament_player_owgr_old_list = [p_owgr for p_owgr in world_player_owgr_old_list if any(player in p_owgr for player in player_list)]
    #print('tournament_player_owgr_old_list:\n', tournament_player_owgr_old_list)
    
    ''' Note: tournament_player_owgr_list = [['Aaraon Baddeley', 253],['Aaron Wise', 54],...,['Zach Johnson', XX]]
        Because the imported tee-time generated player_list has amateurs and older players who don't rank
        in the OWGR anymore, and therefore they're missing in df_owgr, and should be recovered directly from the
        OWGR website. Alternatively, the website could be scraped for every single player in a given tournament
        but it will require selenium to enter the player's name into http://www.owgr.com/about in the player
        profile search bar, clicking his name (<a href="/en/Ranking/PlayerProfile.aspx?playerID=874">Ernie Els</a>)
        and grabbing ranking from the 23rd or 22nd week in 2018.'''
    missing_player_owgr_new_list = [p for p in player_list if not any(player[0] in [p] for player in tournament_player_owgr_new_list)]
    print('missing_player_owgr_new_list:\n', missing_player_owgr_new_list)
    missing_player_owgr_old_list = [p for p in player_list if not any(player[0] in [p] for player in tournament_player_owgr_old_list)]
    print('missing_player_owgr_old_list:\n', missing_player_owgr_old_list)


    ''' Note: Manually adding and subtracting names to the below new and old OWGR lists is tricky.
        Ryan Evans is 301 in the new list,
        but in the old list he shouldn't be included because he rose to 274 on the OWGR, meaning
        he's automatically on the downloaded list. Adding him to the old OWGR list would duplicate
        hist data. '''
    # Manually replacing missing_player_owgr_list:
    missing_player_owgr_new_list = [['Braden Thornberry',797],['Calum Hill',1415],['Cameron Wilson',1416],['Chris Naegel',1098],
                                ['Christopher Babcock',9999],['Chun An Yu',8888],['Cole Miller',1994],['Danny Willett',401],
                                ['David Bransdon',449],['David Gazzolo',1449],['Doug Ghim',1154],['Dylan Meyer',1942],
                                ['Eric Axley',390],['Ernie Els',683],['Franklin Huang',2007],['Garrett Rank',1967],
                                ['Harry Ellis',1986],['Jacob Bergeron',2007],['James Morrison',349],['Kenny Perry',1986],
                                ['Kristoffer Reitan',1342],['Lanto Griffin',328],['Li Haotong',46],['Luis Gagne',1248],
                                ['Matt Jones',302],['Matt Parziale',1958],['Michael Block',1935],['Michael Hebert',750],
                                ['Michael Miller',1093],['Michael Putnam',1053],['Mickey DeMorat',1356],['Noah Goodwin',1995],
                                ['Philip Barbaree',2008],['Rhett Rasmussen',2009],['Rikuya Hoshino',308],['Ryan Evans',301],
                                ['Ryan Lumsden',2010],['Scott Gregory',1294],['Sebastian Munoz',277],['Sebastian Vazquez',1102],
                                ['Shintaro Ban',2011],['Stewart Hagestad',1359],['Sulman Raza',2012],
                                ['Sung Joon Park',1653],['Theo Humphrey',2013],
                                ['Tim Wilkinson',754],['Timothy Wiseman',2014],['Tom Lewis',434],['Ty Strafaci',2015],
                                ['Tyler Duncan',402],['WC Liang',488],['Will Grimmer',1547],['Will Zalatoris',1986]]

    missing_player_owgr_old_list = [['Braden Thornberry',895],['Calum Hill',1415],['Cameron Wilson',1416],['Chris Naegel',1098],
                                ['Christopher Babcock',9999],['Chun An Yu',8888],['Cole Miller',1994],['Danny Willett',401],
                                ['David Bransdon',434],['David Gazzolo',1449],['Doug Ghim',1154],['Dylan Meyer',1942],
                                ['Eric Axley',370],['Ernie Els',632],['Franklin Huang',2007],['Garrett Rank',1967],
                                ['Harry Ellis',1986],['Jacob Bergeron',2007],['James Morrison',346],['Kenny Perry',1986],
                                ['Kristoffer Reitan',1342],['Lanto Griffin',328],['Li Haotong',49],['Luis Gagne',1248],
                                ['Matt Jones',317],['Matt Parziale',1958],['Michael Block',1935],['Michael Hebert',750],
                                ['Michael Miller',1093],['Michael Putnam',1053],['Mickey DeMorat',1356],['Noah Goodwin',1995],
                                ['Philip Barbaree',2008],['Rhett Rasmussen',2009],['Rikuya Hoshino',308],
                                ['Ryan Lumsden',2010],['Scott Gregory',1294],['Sebastian Munoz',291],['Sebastian Vazquez',1102],
                                ['Shintaro Ban',2011],['Stewart Hagestad',1359],['Sulman Raza',2012],
                                ['Sung Joon Park',1653],['Theo Humphrey',2013],
                                ['Tim Wilkinson',754],['Timothy Wiseman',2014],['Tom Lewis',434],['Ty Strafaci',2015],
                                ['Tyler Duncan',402],['WC Liang',488],['Will Grimmer',1547],['Will Zalatoris',1986],
                                ['Shota Akiyoshi',345]]
    
    tournament_player_owgr_new_list.extend(missing_player_owgr_new_list)
    #print('tournament_player_owgr_list:\n', tournament_player_owgr_list)
    tournament_player_owgr_old_list.extend(missing_player_owgr_old_list)

    # Si Woo Kim is Siwoo Kim on OWGR.com
    # Sung Joon Park is Sunjoon Park on OWGR.com
    # Sung-jae Im is Sungjae Im on OWGR.com
    # Ted Potter Jr. is Ted Potter Jr on OWGR.com
    # WC Liang is Liang Wen-Chong on OWGR.com
    
    tournament_player_owgr_new_18_list = tournament_player_owgr_new_list*18
    tournament_player_owgr_old_18_list = tournament_player_owgr_old_list*18

    ''' Dataframe of tournament specific owgr in order to sort the names '''
    
    cols = ['Player','New OWGR']
    df_new = pd.DataFrame(tournament_player_owgr_new_18_list, columns=cols)
    df_new.set_index('Player', inplace=True)
    df_new.sort_index(inplace=True)
    #print('df_new:\n', df_new.to_string())
    cols = ['Player','Old OWGR']
    df_old = pd.DataFrame(tournament_player_owgr_old_18_list, columns=cols)
    df_old.set_index('Player', inplace=True)
    df_old.sort_index(inplace=True)
    #print('df_old:\n', df_old.to_string())

    
    ''' Verifying owgr player list against the imported player list (e.g. player_list) '''
    player_list_owgr_new = df_new.index.get_level_values(0).drop_duplicates().tolist() # List of players only, generated from owgr data
    #print('player_list_owgr_new:\n', player_list_owgr_new)
    player_list_owgr_old = df_old.index.get_level_values(0).drop_duplicates().tolist() # List of players only, generated from owgr data
    #print('player_list_owgr_old:\n', player_list_owgr_old)

    for player_tt, player_owgr_new, player_owgr_old in zip(player_list, player_list_owgr_new, player_list_owgr_old):
        if player_tt != player_owgr_new or player_tt != player_owgr_old:
            print('########  WARNING: Tee time player', player_tt, 'name doesn\'t match new OWGR name', player_owgr_new,'or old OWGR name', player_owgr_old)
    # Making a dataframe of both player name lists to make sure they're the same:
    df_players = pd.DataFrame(player_list, columns=['TT Names'])
    df_players.insert(loc=1, column='New OWGR Names', value=player_list_owgr_new)
    df_players.insert(loc=2, column='Old OWGR Names', value=player_list_owgr_old)
    
    print('################# df_players: All columns should be the same:\n', df_players.to_string())
    
    ''' Pickle out tournament specific owgr '''
    df_owgr_new = df_new
    df_owgr_new.to_pickle('us_open_owgr_new.pickle')

    df_owgr_old = df_old
    df_owgr_old.to_pickle('us_open_owgr_old.pickle')
    

    ''' Pickle out entire world's players' owgr '''
    df_world_owgr_new = pd.DataFrame(world_player_owgr_new_list, columns=cols)
    df_world_owgr_new.to_pickle('world_owgr_new.pickle')
    ''' Pickle out entire world's players' owgr '''
    df_world_owgr_old = pd.DataFrame(world_player_owgr_old_list, columns=cols)
    df_world_owgr_old.to_pickle('world_owgr_old.pickle')
    

    ''' Making old-new difference dataframe: '''
    player_sorted_18_list = df_new.index.values.tolist() # Getting names repeated 18 times
    owgr_ranking_new_list = df_new['New OWGR'].get_values().tolist() # Getting New OWGR ranking
    owgr_ranking_old_list = df_old['Old OWGR'].get_values().tolist() # Getting Old OWGR ranking
    owgr_ranking_diff_list = list(map(operator.sub, owgr_ranking_new_list, owgr_ranking_old_list)) # Subtracting Old From new OWGR
    player_owgr_old_new_diff_list = [[w,x,y,z] for w,x,y,z in zip(player_sorted_18_list, owgr_ranking_old_list, owgr_ranking_new_list, owgr_ranking_diff_list)]
    
    df_owgr = pd.DataFrame(player_owgr_old_new_diff_list, columns=['Player','Old OWGR','New OWGR','OWGR Diff'])
    df_owgr.dropna(inplace=True)
    print('df_owgr:\n', df_owgr.to_string())

    ''' Pickle out OWGR '''
    df_owgr.to_pickle('us_open_owgr.pickle')
    
    return df_owgr # ,df_owgr_new, df_owgr_old, df_world_owgr_new, df_world_owgr_old
    









'''################################################################################################'''
# This function calculates the round progression of each player's score to the minute
def GetRoundProgress(df_scorecards, df_tee_times, df_owgr, year_list, date_list):
    print('year_list:\n', year_list)
    print('date_list:\n', date_list)
    print('df_tee_times:\n', df_tee_times)
    print('df_scorecards:\n', df_scorecards)

    ########  Creating a lookup table for score progressions  ########
    # Importing tab-separated hole duration numbers:
    data_list = []
    with open('2018_us_open_hole_dur_lookup.csv') as tsv:
        #You can also use delimiter="\t" rather than giving a dialect.
        for line in csv.reader(tsv, dialect='excel-tab'):
            line = list(line[0].split(','))[0:5] # [0:5] avoids the time conversion number after the 4th element
            if line[0].isdigit(): # If line[0] is a digit, line[1] and line[2] are digits as well.
                line[0] = int(line[0]) # First through third elements are 
                line[1] = int(line[1]) # strings of group size, par and to par
                line[2] = int(line[2]) # numbers. Converting to ints.
            data_list.append(line)
    cols = data_list[0] # hole_duration_list = [['Group','Par','To Par','Time'],['3','-2','0:08:00'],['3','-1',...]]
    data_list = data_list[1:] # Skipping the title headers
    df_hole_dur_lookup = pd.DataFrame(data=data_list, columns=cols)
    #print('type(df_hole_dur_lookup.iloc[10][\'Start\']):\n', type(df_hole_dur_lookup.iloc[10]['Start']))

    # Converting df lookup table Start and End column data to Timestamp objects:
    start_list = [dt.datetime.strptime(t,'%H:%M:%S') for t in df_hole_dur_lookup['Start']]
    end_list = [dt.datetime.strptime(t,'%H:%M:%S') for t in df_hole_dur_lookup['End']]
    hole_duration_list = list(map(operator.sub, end_list, start_list))
    print('hole_duration_list[0:10]:\n', hole_duration_list[0:10])
    print('type(hole_duration_list[10]:\n', type(hole_duration_list[10]))

    #return df_scorecards, df_scorecards, df_scorecards
########  OLD CODE, DON'T NEED:
##    df_hole_dur_lookup['Start'] = pd.to_datetime(df_hole_dur_lookup['Start'], format='%H:%M:%S')
##    df_hole_dur_lookup['End'] = pd.to_datetime(df_hole_dur_lookup['End'], format='%H:%M:%S')
##    print('type(df_hole_dur_lookup.iloc[10][\'End\']):', type(df_hole_dur_lookup.iloc[10]['End']))
##    start_list = df_hole_dur_lookup['Start'].values.tolist()
##    end_list = df_hole_dur_lookup['End'].values.tolist()
##    # Converting df's Timestamp objects to datetime objects for later addition:
##    start_list = [i.to_datetime() for i in df_hole_dur_lookup['Start']]
##    end_list = [i.to_datetime() for i in df_hole_dur_lookup['End']]
##    hole_duration_list = list(map(operator.sub, end_list, start_list))
########
    
    # Drop Start and End columns, replace with Time column containin timedelta objects
    df_hole_dur_lookup.drop(columns=['Start','End'], inplace=True)
    df_hole_dur_lookup.insert(loc=3, column='Time', value=hole_duration_list)
    df_hole_dur_lookup.set_index(['Group Size','Par','To Par'], inplace=True)
    print('df_hole_dur_lookup:\n', df_hole_dur_lookup)
    print('type(df_hole_dur_lookup.iloc[10][\'Time\']):\n', type(df_hole_dur_lookup.iloc[10]['Time']))
    #return df_hole_dur_lookup, df_hole_dur_lookup, df_hole_dur_lookup
    #################################################################

    ########  Importing df's with PickleInPlayersScorecardsAndTeeTimes()  ########
    # Inserting into df_scorecards a column with hole played for each player = 'Hole'. This can't be
    # done ahead of time because ESPN doesn't provide tee-off hole. Alternatively, use tee times
    # during scorecard collection to assign holes to scores.

    # df scorecards gets players based on where they finished in the tournament
    # Thus, Brooks Koepka appears first
    # df tee times gets players based on tee time list, thus players
    # are organized according to ability
    # To address this, I am reordering all lists by first name alphabetically

    ########  THIS ULTIMATELY NEEDS TO BE MOVED INTO GetScorecardsESPN()  ########
    # Setting df_scorecards indices as Player and Temp Index (integer sequence)
    # Is this the right way to append a sequence of numbers to a dataframe?
    # I'm making a list, converting to a numpy array, then writing it in
    # without using assign():
    rows = len(df_scorecards.index)
    # This temp index keeps each hole's score in order when index is set to
    # 'Player', otherwise the player names are sorted alphabetically while
    # hole order is lost.
    np_temp_index = np.asarray(list(range(0,rows)))
    #print('np_temp_index:\n', np_temp_index)
    df_scorecards['Temp Index'] = np_temp_index
    df_scorecards.set_index(['Player','Temp Index'], inplace=True)
    df_scorecards.sort_index(level='Player', sort_remaining=True, inplace=True)
    print('df_scorecards:\n', df_scorecards)
    #print('df_scorecards.loc[\'Charl Schwartzel\']:\n', df_scorecards.loc['Charl Schwartzel'])
    ###############################################################################

    #return df_scorecards, df_tee_times, df_tee_times

    ########  Sorting out the non-duplicate player lists from df_tee_times and df_scorecards  ########
    # I don't have a player list, stepping through all players in df_tee_times
    # Getting from tee times because I'll be searching tee times and need the
    # name spellings to match:
    player_list_teetimes = df_tee_times.index.get_level_values(0).drop_duplicates().tolist() # Dropping duplicates 
    player_list_scorecards = df_scorecards.index.get_level_values(0).drop_duplicates().tolist() # Dropping duplicates 
    # Comparing both tee time and scorecard name lists to ensure that hole list data
    # originating from df_tee_time can be copied directly to df_scorecards
    # and the data will find the right player because both df's have the same player order
    for player_tt, player_sc in zip(player_list_teetimes, player_list_scorecards):
        if player_tt != player_sc:
            print('########  WARNING: Tee time player', player_tt, 'name doesn\'t match scorecard name', player_sc)
    # Making a dataframe of both player name lists to make sure they're the same:
    df_players = pd.DataFrame(player_list_teetimes, columns=['TT Names'])
    df_players.insert(loc=1, column='SC Names', value=player_list_scorecards)
    print('#################  df_players: Both should be the same:\n', df_players.to_string()) # .to_string() prints the entire dataframe for visual comparison
    ##################################################################################################

    # Making lists based on tee-off hole:
    r1=range(1,10); r10=range(10,19) # range 1-8 and range 10-18
    hole_list_1 = [*r1,*r10]; hole_list_10 = [*r10,*r1]
    ########  PAR FOR 2018 US OPEN ONLY, CHANGE FOR OTHER TOURNAMENTS:
    par_list_1 = [4,3,4,4,5,4,3,4,4,4,3,4,4,4,4,5,3,4]
    par_list_10 = [4,3,4,4,4,4,5,3,4,4,3,4,4,5,4,3,4,4]
    pin_list_round_1_hole_1 =   ['BR','CR','BL','CL','CL','CC','FC','FL','CC', 'BC','CL','FR','CR','BC','FC','CC','BR','BL']
    pin_list_round_1_hole_10 =  ['BC','CL','FR','CR','BC','FC','CC','BR','BL', 'BR','CR','BL','CL','CL','CC','FC','FL','CC']
    pin_list_round_2_hole_1 =   ['BL','BL','CL','CR','FR','CC','CL','BR','CC', 'FC','CC','BL','BC','CL','BC','FR','CL','CC']
    pin_list_round_2_hole_10 =  ['FC','CC','BL','BC','CL','BC','FR','CL','CC', 'BL','BL','CL','CR','FR','CC','CL','BR','CC']
    pin_list_round_3 =          ['FL','FC','FR','BL','CR','CL','FR','CL','CL', 'CL','CR','FC','CR','BR','CR','BC','FC','BL']
    pin_list_round_4 =          ['CC','BC','CC','FR','BC','BL','FL','CC','CL', 'CR','BR','CL','CC','FC','CL','CC','BL','CC']
    
    time_dur_list = []
    nan_list = [np.NaN]*18
    #print('tee_off_list:\n', tee_off_list)
    #print('len(tee_off_list):',len(tee_off_list),'should equal len(player_list)/4:', len(player_list)/4)
    # I'M PULLING EACH COLUMN FROM DF_TEE_TIMES WHICH DOESN'T SEEM EFFICIENT:
    player_list = df_tee_times.index.get_level_values(0).tolist() # Has 4 duplicates, this is needed for loop
    round_list = df_tee_times.index.get_level_values(1).tolist()
    tee_off_list = df_tee_times.index.get_level_values(2).tolist()
    group_size_list = df_tee_times.index.get_level_values(3).tolist()
    group_id_list = df_tee_times.index.get_level_values(4).tolist() # NOT DOING ANYTHING WITH THIS YET
    tee_time_list = df_tee_times['Tee Time'].tolist()
    #print('df_tee_times:\n', df_tee_times)
    #print('df_tee_times[\'Tee Time\']:\n', df_tee_times['Tee Time'])
    print('tee_time_list[0:20]:\n', tee_time_list[0:20])
    print('type(tee_time_list[0]): ', type(tee_time_list[0]))

    round_1_scores_list = []; round_2_scores_list = []; round_3_scores_list = []; round_4_scores_list = []
    round_1_hole_list = []; round_2_hole_list = []; round_3_hole_list = []; round_4_hole_list = []
    round_1_pin_list = []; round_2_pin_list = []; round_3_pin_list = []; round_4_pin_list = []
    round_1_par_list = []; round_2_par_list = []; round_3_par_list = []; round_4_par_list = []
    round_1_to_par_list = []; round_2_to_par_list = []; round_3_to_par_list = []; round_4_to_par_list = []
    round_1_to_par_cumsum_list = []; round_2_to_par_cumsum_list = []; round_3_to_par_cumsum_list = []; round_4_to_par_cumsum_list = []
    round_1_time_dur_list = []; round_2_time_dur_list = []; round_3_time_dur_list = []; round_4_time_dur_list = [] 
    round_1_time_cumsum_list = []; round_2_time_cumsum_list = []; round_3_time_cumsum_list = []; round_4_time_cumsum_list = []
    # Using player list from scorecards because scores data is pulled. And while Alex Noren
    # exists in df_tee_times he exists as Alexander Noren in df_scorecards
    # e.g. player, rnd, hole:
    # 'Aaron Baddeley',Round 1','10' followed by
    # 'Aaron Baddeley','Round 2','1'
    i = 0
    print(df_hole_dur_lookup)

    # Note that group_size is grabbed from a df_tee_times derived list here, which is then used alongside
    # df_scorecard derived par and to-par several loops nested to look-up values in df_hole_dur_lookup.
    for player, rnd, hole, grp_size, tee_time in zip(player_list, round_list, tee_off_list, group_size_list, tee_time_list):
        #print('Player:', player)
        progress = math.ceil(i/len(player_list)*100)
        print('Progress: ', progress,'%')
        if rnd == 'Round 1':
            #print('Round 1')
            # If the Round 1 column of Player has any NaNs, set lists to all NaNs
            # because we I can't search the lookup table with to_par values that are NaNs
            if df_scorecards.loc[player][rnd].isnull().values.any() == True:
                # Recall that some players have no scores for all four rounds due to a score
                # collection problem in GetScorecardsESPN():
                print('###################  Round 1 NaN scores present')
                round_1_scores_list.extend(nan_list)
                round_1_hole_list.extend(nan_list)
                round_1_pin_list.extend(nan_list)
                round_1_par_list.extend(nan_list)
                round_1_to_par_list.extend(nan_list)
                round_1_to_par_cumsum_list.extend(nan_list)
                round_1_time_dur_list.extend(nan_list)
                round_1_time_cumsum_list.extend(nan_list)
            else:
                scores_list = df_scorecards.loc[player][rnd].values.tolist()
                #print('round 1 scores_list for', player,':\n', scores_list)
                if hole == '1': # These should be converted to ints in GetTeeTimes()
                    round_1_scores_list.extend(scores_list) # Keep scores list as it is
                    hole_list = hole_list_1 # Writing to generic hole list makes referencing holes easier
                    round_1_hole_list.extend(hole_list)
                    pin_list = pin_list_round_1_hole_1
                    round_1_pin_list.extend(pin_list)
                    par_list = par_list_1 # Writing to generic par list makes referencing par easier in for loop below
                    round_1_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list)) # Subtract lists: -1 in list means birdie
                    round_1_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_1_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = [] # Need to initialize here
                    time_cumsum_list = []
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        #print('Round 1: group_size =', grp_size,'par =',par,', to par =',to_par)
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        #print('time_dur:', time_dur)
                        #print('type(time_dur):', type(time_dur))
                        time_dur_list.extend(time_dur)
                        #print('Round 1 time_dur_list:\n', time_dur_list)
                        if len(time_cumsum_list) == 0: 
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    if len(time_cumsum_list) != 18:
                        print('#########  WARNING: Length of time_cumsum_list=',len(time_cumsum_list),'!= 18:\n')
                        print('#########  time_cumsum_list:\n', time_cumsum_list)
                    round_1_time_dur_list.extend(time_dur_list)
                    round_1_time_cumsum_list.extend(time_cumsum_list)
    
                elif hole == '10':
                    scores_list = scores_list[9:18]+scores_list[0:9]
                    round_1_scores_list.extend(scores_list)
                    hole_list = hole_list_10
                    round_1_hole_list.extend(hole_list)
                    pin_list = pin_list_round_1_hole_10
                    round_1_pin_list.extend(pin_list)
                    par_list = par_list_10
                    round_1_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list))
                    round_1_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_1_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = []
                    time_cumsum_list = []
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        time_dur_list.extend(time_dur)
                        if len(time_cumsum_list) == 0: 
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    round_1_time_dur_list.extend(time_dur_list)
                    round_1_time_cumsum_list.extend(time_cumsum_list)
                    
        elif rnd == 'Round 2':
            #print('Round 2')
            if df_scorecards.loc[player][rnd].isnull().values.any() == True:
                print('###################  Round 2 NaN scores present')
                round_2_scores_list.extend(nan_list)
                round_2_hole_list.extend(nan_list)
                round_2_pin_list.extend(nan_list)
                round_2_par_list.extend(nan_list)
                round_2_to_par_list.extend(nan_list)
                round_2_to_par_cumsum_list.extend(nan_list)
                round_2_time_dur_list.extend(nan_list)
                round_2_time_cumsum_list.extend(nan_list)
            else:
                scores_list = df_scorecards.loc[player][rnd].values.tolist()
                if hole == '1':
                    print('hole = 1')
                    round_2_scores_list.extend(scores_list)
                    hole_list = hole_list_1
                    round_2_hole_list.extend(hole_list)
                    pin_list = pin_list_round_2_hole_1
                    round_2_pin_list.extend(pin_list)
                    par_list = par_list_1
                    round_2_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list))
                    round_2_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_2_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = []
                    time_cumsum_list = [] # Initialize with tee_time
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        time_dur_list.extend(time_dur)
                        if len(time_cumsum_list) == 0:
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    round_2_time_dur_list.extend(time_dur_list)
                    round_2_time_cumsum_list.extend(time_cumsum_list)
                elif hole == '10':
                    print('hole = 10')
                    scores_list = scores_list[9:18]+scores_list[0:9]
                    round_2_scores_list.extend(scores_list)
                    hole_list = hole_list_10
                    round_2_hole_list.extend(hole_list)
                    pin_list = pin_list_round_2_hole_10
                    round_2_pin_list.extend(pin_list)
                    par_list = par_list_10
                    round_2_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list))
                    round_2_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_2_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = []
                    time_cumsum_list = [] # Initialize with tee_time
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        time_dur_list.extend(time_dur)
                        if len(time_cumsum_list) == 0:
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    round_2_time_dur_list.extend(time_dur_list)
                    round_2_time_cumsum_list.extend(time_cumsum_list)

        elif rnd == 'Round 3':
            #print('Round 3')
            if df_scorecards.loc[player][rnd].isnull().values.any() == True:
                print('###################  Round 3 NaN scores present')
                round_3_scores_list.extend(nan_list)
                round_3_hole_list.extend(nan_list)
                round_3_pin_list.extend(nan_list)
                round_3_par_list.extend(nan_list)
                round_3_to_par_list.extend(nan_list)
                round_3_to_par_cumsum_list.extend(nan_list)
                round_3_time_dur_list.extend(nan_list)
                round_3_time_cumsum_list.extend(nan_list)
            else:
                scores_list = df_scorecards.loc[player][rnd].values.tolist()
                if hole == '1':
                    round_3_scores_list.extend(scores_list)
                    hole_list = hole_list_1
                    round_3_hole_list.extend(hole_list)
                    pin_list = pin_list_round_3
                    round_3_pin_list.extend(pin_list)
                    par_list = par_list_1
                    round_3_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list))
                    round_3_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_3_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = []
                    time_cumsum_list = [] # Initialize with tee_time
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        time_dur_list.extend(time_dur)
                        if len(time_cumsum_list) == 0:
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    round_3_time_dur_list.extend(time_dur_list)
                    round_3_time_cumsum_list.extend(time_cumsum_list)

        elif rnd == 'Round 4':
            #print('Round 4')
            if df_scorecards.loc[player][rnd].isnull().values.any() == True:
                print('###################  Round 4 NaN scores present')
                round_4_scores_list.extend(nan_list)
                round_4_hole_list.extend(nan_list)
                round_4_pin_list.extend(nan_list)
                round_4_par_list.extend(nan_list)
                round_4_to_par_list.extend(nan_list)
                round_4_to_par_cumsum_list.extend(nan_list)
                round_4_time_dur_list.extend(nan_list)
                round_4_time_cumsum_list.extend(nan_list)
            else:
                scores_list = df_scorecards.loc[player][rnd].values.tolist()
                if hole == '1':
                    round_4_scores_list.extend(scores_list)
                    hole_list = hole_list_1
                    round_4_hole_list.extend(hole_list)
                    pin_list = pin_list_round_4
                    round_4_pin_list.extend(pin_list)
                    par_list = par_list_1
                    round_4_par_list.extend(par_list)
                    to_par_list = list(map(operator.sub, scores_list, par_list))
                    round_4_to_par_list.extend(to_par_list)
                    to_par_cumsum_list = list(accumulate(to_par_list))
                    round_4_to_par_cumsum_list.extend(to_par_cumsum_list)
                    time_dur_list = []
                    time_cumsum_list = [] # Initialize with tee_time
                    for par, to_par in zip(par_list, to_par_list): # Compiling time durations
                        time_dur = df_hole_dur_lookup.loc[grp_size,par,to_par]
                        time_dur_list.extend(time_dur)
                        if len(time_cumsum_list) == 0:
                            new_time = tee_time + time_dur
                        else: # If more than the tee time is in the list
                            new_time += time_dur
                        time_cumsum_list.extend(new_time)
                    round_4_time_dur_list.extend(time_dur_list)
                    round_4_time_cumsum_list.extend(time_cumsum_list)

        i += 1

    player_list_scorecards = df_scorecards.index.get_level_values(0).tolist()    

    print('len(round_4_scores_list):', len(round_4_scores_list))
    print('len(round_4_hole_list):', len(round_4_hole_list))
    print('len(round_4_pin_list):', len(round_4_pin_list))
    print('len(round_4_par_list):', len(round_4_par_list))
    print('len(round_4_to_par_list):', len(round_4_to_par_list))
    print('len(round_4_to_par_cumsum_list):', len(round_4_to_par_cumsum_list))
    print('len(round_4_time_dur_list):', len(round_4_time_dur_list))
    print('len(round_4_cumsum_list):', len(round_4_time_cumsum_list))
    
    # Making a new scorecards df:
    df = pd.DataFrame(player_list_scorecards, columns=['Player'])

    owgr_new_list = df_owgr['New OWGR'].tolist()
    df.insert(loc=1, column='New OWGR', value=owgr_new_list)

    owgr_diff_list = df_owgr['OWGR Diff'].tolist()
    df.insert(loc=2, column='OWGR Diff', value=owgr_diff_list)
    
    df.insert(loc=3, column='R1 Time', value=round_1_time_cumsum_list)
    df.insert(loc=4, column='R2 Time', value=round_2_time_cumsum_list)
    df.insert(loc=5, column='R3 Time', value=round_3_time_cumsum_list)
    df.insert(loc=6, column='R4 Time', value=round_4_time_cumsum_list)

    df.insert(loc=7, column='R1 Hole', value=round_1_hole_list)
    df.insert(loc=8, column='R2 Hole', value=round_2_hole_list)
    df.insert(loc=9, column='R3 Hole', value=round_3_hole_list)
    df.insert(loc=10, column='R4 Hole', value=round_4_hole_list)

    df.insert(loc=11, column='R1 To Par', value=round_1_to_par_list)
    df.insert(loc=12, column='R2 To Par', value=round_2_to_par_list)
    df.insert(loc=13, column='R3 To Par', value=round_3_to_par_list)
    df.insert(loc=14, column='R4 To Par', value=round_4_to_par_list)
    
    df.insert(loc=15, column='R1 To Par CS', value=round_1_to_par_cumsum_list)
    df.insert(loc=16, column='R2 To Par CS', value=round_2_to_par_cumsum_list)
    df.insert(loc=17, column='R3 To Par CS', value=round_3_to_par_cumsum_list)
    df.insert(loc=18, column='R4 To Par CS', value=round_4_to_par_cumsum_list)

    df.insert(loc=19, column='R1 Time Dur', value=round_1_time_dur_list)
    df.insert(loc=20, column='R2 Time Dur', value=round_2_time_dur_list)
    df.insert(loc=21, column='R3 Time Dur', value=round_3_time_dur_list)
    df.insert(loc=22, column='R4 Time Dur', value=round_4_time_dur_list)

    df.insert(loc=23, column='R1 Pin', value=round_1_pin_list)
    df.insert(loc=24, column='R2 Pin', value=round_2_pin_list)
    df.insert(loc=25, column='R3 Pin', value=round_3_pin_list)
    df.insert(loc=26, column='R4 Pin', value=round_4_pin_list)
    
    df['R1 Time Start'] = df['R1 Time'] - df['R1 Time Dur']
    df['R2 Time Start'] = df['R2 Time'] - df['R2 Time Dur']
    df['R3 Time Start'] = df['R3 Time'] - df['R3 Time Dur']
    df['R4 Time Start'] = df['R4 Time'] - df['R4 Time Dur']

    ''' Preparing tee times for df_round_progress_final_score: '''
    round_1_tee_time_list = tee_time_list[0::4]
    round_2_tee_time_list = tee_time_list[1::4]
    round_3_tee_time_list = tee_time_list[2::4]
    round_4_tee_time_list = tee_time_list[3::4]
    print('round_4_tee_time_list:\n', round_4_tee_time_list)
    
    # Overwrite df_scorecards with a newly constructed dataframe with correct
    # score orders and their corresponding to-par scores:
    df_round_progress = df
    df_round_progress.set_index('Player', inplace=True)



    ''' Creating dataframe of each round's final score: '''
    master_final_score_list = []
    # player_list from scorecards has duplicate names, using df index to avoid duplicates:
    player_index_list = df_round_progress.index.drop_duplicates().values.tolist()
    for player in player_index_list:
        final_score_series = df_round_progress.loc[player].iloc[-1] # Reading one row, all columns into df_round_progress_final_score
        final_score_list = [player] + final_score_series.values.tolist()
        master_final_score_list.append(final_score_list)
    cols = ['Player'] + df.columns.tolist()
    df_round_progress_final_score = pd.DataFrame(master_final_score_list, columns=cols)

    ''' Inserting tee times into df_round_progress_final_score: '''
    df_round_progress_final_score.insert(loc=1, column='R1 Tee Time', value=round_1_tee_time_list)
    df_round_progress_final_score.insert(loc=2, column='R2 Tee Time', value=round_2_tee_time_list)
    df_round_progress_final_score.insert(loc=3, column='R3 Tee Time', value=round_3_tee_time_list)
    df_round_progress_final_score.insert(loc=4, column='R4 Tee Time', value=round_4_tee_time_list)

    df_round_progress_final_score.set_index('Player', inplace=True)



    # Send to pickle:
    df_round_progress.to_pickle('us_open_round_progress.pickle')
    df_round_progress_final_score.to_pickle('us_open_round_progress_final_score.pickle')

    print('df_round_progress:\n', df_round_progress)
    print('df_round_progress_final_score\n', df_round_progress_final_score)
    
    return df_round_progress, df_round_progress_final_score, df_scorecards, df_tee_times, df_owgr



''' Round Progress EDA: Plots rolling averages of hole scores for each day '''
def RoundProgressEDA(df_round_progress):

    dfs = pd.DataFrame()
    for x in ['R1','R2','R3','R4']:
        time_str = x + ' Time'
        to_par_str = x + ' To Par'
        df = df_round_progress[[time_str,to_par_str]]
        df.reset_index(inplace=True)
        df.rename({time_str:'Time', to_par_str:'To Par'}, axis='columns', inplace=True)
        df.set_index('Time', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        print('df:\n', df)
        
        dfs = dfs.append(df)
  
    df_rolling_30m_avg = dfs.rolling('1800s', min_periods=30).mean()
    df_rolling_30m_avg.rename({'To Par':'30 min rolling avg'}, axis='columns', inplace=True)
    df_rolling_1hr_avg = dfs.rolling('3600s', min_periods=60).mean()
    df_rolling_1hr_avg.rename({'To Par':'1 hr rolling avg'}, axis='columns', inplace=True)
    df_rolling_2hr_avg = dfs.rolling('7200s', min_periods=120).mean()
    df_rolling_2hr_avg.rename({'To Par':'2 hr rolling avg'}, axis='columns', inplace=True)


    plt.figure()
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['30 min rolling avg'],c='g',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['1 hr rolling avg'],c='k',alpha=0.7)
    #plt.plot(df_rolling_2hr_avg.index, df_rolling_2hr_avg['2 hr rolling avg'],c='k',alpha=0.7)

    plt.xlabel('DateTime'); plt.ylabel('Score Rel. to Par')
    plt.legend(loc='lower left')

    fig = plt.gcf()
    fig.set_size_inches(7,3) # 5 wide by 3 high
    fig.savefig('2018 US Open - Rolling Averages - Hole Scores.png', bbox_inches='tight')
    
    plt.show()

    return dfs



    



''' Match df_round_progress to the weather '''
def MakeTrainingSet(df_proc_imp_weather, df_round_progress, df_round_progress_final_score, year_list, date_list):

    ## NOT USING THIS BUT MAY BE USEFUL FOR IMPUTING MINUTE-RESOLUTION SCORE DATA
##    ########  Making a minute-to-minute datetime objects list for all tournament days  ########
##    num_days = len(date_list)
##    a = date_list
##    #[element y for list x in list a for element y in [thing you want to make with x]]
##    date_list_subhourly = [y for x in a for y in [x]*1440]
##    a = list(range(0,24))*num_days
##    hr_list_subhourly = [y for x in a for y in [str(x)]*60] # How to read this? Creates [0x59,1X59,...,23X59]
##    a = list(range(0,60))*24*num_days
##    minutes_list_subhourly = [y for x in a for y in [str(x)]]
##    time_list_subhourly = [y for i,j,k in zip(date_list_subhourly, hr_list_subhourly, minutes_list_subhourly) for y in [i+' '+j+':'+k+':00']]
##
##    datetime_list_subhourly = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in time_list_subhourly]
##    ###########################################################################################

    #df_proc_imp_weather.reset_index(inplace=True)

    player_list = df_round_progress.index.get_level_values(0).drop_duplicates().tolist()
    #######  Should rename df_round_progress columns to R1 Time End, etc
    rX_tee_time_list = ['R1 Tee Time', 'R2 Tee Time', 'R3 Tee Time', 'R4 Tee Time']
    rX_time1_list = ['R1 Time Start', 'R2 Time Start', 'R3 Time Start', 'R4 Time Start']
    rX_time2_list = ['R1 Time', 'R2 Time', 'R3 Time', 'R4 Time']
    rX_hole_list = ['R1 Hole', 'R2 Hole', 'R3 Hole', 'R4 Hole']
    rX_pin_list = ['R1 Pin', 'R2 Pin', 'R3 Pin', 'R4 Pin']
    rX_to_par_cs_list = ['R1 To Par CS', 'R2 To Par CS', 'R3 To Par CS', 'R4 To Par CS']
    rX_to_par_list = ['R1 To Par', 'R2 To Par', 'R3 To Par', 'R4 To Par']

    ''' Making df_training_set: contains individual hole data '''
    df_training_set = pd.DataFrame() # Will contain data for each hole
    n = len(player_list)
    for i,player in enumerate(player_list):
        p = round((i+1)/n*100, 1)
        print('Progress:', p,'%')
        
        ''' For every player, the datetime, score and owgr must be grabbed from
            df_round_progress. There may be a way to do this without iteration. '''
        for rtt, rt1, rt2, rh, rp, rtp in zip(rX_tee_time_list, rX_time1_list, rX_time2_list, rX_hole_list, rX_pin_list, rX_to_par_list):
            # Dataframe to merge weather to: df = special selection of df_round_progress:
            df = df_round_progress.loc[player][[rt2, rh, rp, rtp, 'New OWGR', 'OWGR Diff']] # Don't need rt1. rcs is round cumsum from when rX_to_par_cs_list was used instead of rX_to_par_list
            df.reset_index(inplace=True)
            df.rename(columns={rt2:'DateTime', rh:'Hole', rp:'Pin', rtp:'Score'}, inplace=True)

            ''' MATCHING ROUND PROGRESS DATA TO WEATHER DATA: '''
            ''' Get df_proc_imp_weather data for time range from hole time recorded
                in df_round_progress to time_dur prior to that time.
                Average and std dev of the weather during that time.
                Store those weather stats the hole's timestamp in df_proc_imp_weather. '''
            
            weather_list = []
            #######  WARNING: Currently converting pandas series df_mean to a list, then
            # append that list, extracting the series row names, and building a dataframe
            # out of row names and the list. There must be a more efficient way of
            # combining pandas series.
            ''' We're two loops deep. We're iterating through every player and every round,
                but to get the weather for each hole, we need to iterate through each hole,
                e.g. each row, which is what is being done here: '''
            for t1, t2 in zip(df_round_progress.loc[player][rt1], df_round_progress.loc[player][rt2]):
                df_mean = df_proc_imp_weather.loc[t1:t2].mean(axis=0)
                #df_sd = df_proc_imp_weather.loc[t1:t2].std()
                mean_list = df_mean.values.tolist()
                time_mean_list = [t2] + mean_list # Adding timestamp of hole finish to list of averaged weather.
                weather_list.append(time_mean_list) # Appends for every player's hole completed
            cols_weather = ['DateTime'] + df_mean.index.values.tolist() # Index values are column names because df_mean is a series
            df_weather = pd.DataFrame(weather_list, columns=cols_weather)
            df_merged = df.merge(df_weather, how='left', on='DateTime')
            # df_training_set contains scores and weather for each hole's time duration
            df_training_set = df_training_set.append(df_merged)


    ''' Making df2_training_set: contains individual round data: '''
    df2_training_set = pd.DataFrame() # Will contain data for each round
    # Hole list (rX_hole_list) and pin list (rX_pin_list) are left out of the iteration
    # because there's no need to keep track of each hole.
    player_list = df_round_progress_final_score.index.drop_duplicates().values.tolist()
    n = len(player_list)
    round_list = [1,2,3,4]*n # For insertion into df2_training_set once it is created (see below for loop)
    for rtt, rt2, rcs in zip(rX_tee_time_list, rX_time2_list, rX_to_par_cs_list):
        # Dataframe to merge weather to: df2 = special selection of df_round_progress_final_score:
        df2 = df_round_progress_final_score[[rtt, rt2, rcs, 'New OWGR', 'OWGR Diff']]
        df2.reset_index(inplace=True)
        df2.rename(columns={player: 'Player', rtt: 'Tee Time', rt2:'DateTime', rcs:'Score'}, inplace=True)
        #print('df2:\n', df2.to_string())
        #print('len(df2:\n', len(df2))
        #print('len(player_list):', len(player_list))
        
        weather_list = []
        ''' Producing weather df by iterating through each round's Tee Time and Time columns: '''
        for tt, t2 in zip(df_round_progress_final_score.iloc[:][rtt], df_round_progress_final_score.iloc[:][rt2]):
            df_mean = df_proc_imp_weather.loc[tt:t2].mean(axis=0)
            #df2_sd = df_proc_imp_weather.loc[tt:t2].std()
            mean_list = df_mean.values.tolist()
            time_mean_list = [t2] + mean_list # Adding timestamp of round finish to list of averaged weather.
            weather_list.append(time_mean_list) # Appends for every player's hole completed
        cols_weather = ['DateTimeWeather'] + df_mean.index.values.tolist()
        df_weather = pd.DataFrame(weather_list, columns=cols_weather)
        #print('df_weather:\n', df_weather.to_string())
        #print('len(df_weather):\n', len(df_weather))
        ''' Not merging. df2 and df_weather align, putthing them together with concat: '''
        df_merged = pd.concat([df2, df_weather], axis=1)
        #print('df_merged:\n', df_merged.to_string())
        #print('len(df_merged):', len(df_merged))
   
        # df_training_set contains scores and weather for each hole's time duration
        df2_training_set = df2_training_set.append(df_merged)

    ############  Should write a warning in the event DateTime values from df2 and df_weather aren't equal
    print('Make sure DateTime and DateTimeWeather values are equal')
    df2_training_set.set_index(['Player','DateTime'], inplace=True)
    df2_training_set.sort_index(inplace=True)
    df2_training_set.insert(loc=1, column='Round', value=round_list)

    ''' Removing df_training_set rows that contain
        NaNs for cut rounds or bad score collection: '''
    df_training_set.dropna(inplace=True)
    df2_training_set.dropna(inplace=True)

    ''' Changing Hole and OWGR types to int: '''
    df_training_set['Hole'] = df_training_set['Hole'].astype(int)
    df_training_set['New OWGR'] = df_training_set['New OWGR'].astype(int)
    df_training_set['OWGR Diff'] = df_training_set['OWGR Diff'].astype(int)
    
    df2_training_set['Round'] = df2_training_set['Round'].astype(int)
    df2_training_set['New OWGR'] = df2_training_set['New OWGR'].astype(int)
    df2_training_set['OWGR Diff'] = df2_training_set['OWGR Diff'].astype(int)

    
    # Putting Score column at far right
    df_scores = df_training_set.pop('Score')
    df_training_set['Score'] = df_scores
    df_training_set['Score'] = df_training_set['Score'].astype(int)

    df_scores = df2_training_set.pop('Score')
    df2_training_set['Score'] = df_scores
    df2_training_set['Score'] = df2_training_set['Score'].astype(int)
    

    df_training_set.to_pickle('us_open_training_set.pickle')
    df2_training_set.to_pickle('us_open_training_set_2.pickle')
    print('df_training_set:\n', df_training_set)
    print('df2_training_set:\n', df2_training_set)

    return df_training_set, df2_training_set




''' Training Set EDA: '''
def TrainingSetEDA(df_training_set):

    df = df_training_set.drop(columns=['Player','Hole'])
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print('df:\n', df)

    ''' Plotting Training Set features '''
    df_rolling_30m_avg = df.rolling('1200s', min_periods=20).mean()
    df_rolling_1hr_avg = df.rolling('3600s', min_periods=60).mean()
    df_rolling_2hr_avg = df.rolling('7200s', min_periods=120).mean()

    plt.figure()
    
    plt.subplot(3,2,1)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['New OWGR'],c='orange',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['New OWGR'],c='k',alpha=0.7)
    plt.ylabel('OWGR'); plt.xticks(rotation=45); plt.xticks(rotation=45)
    frame1 = plt.gca(); frame1.axes.xaxis.set_ticklabels([])

    plt.subplot(3,2,2)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['OWGR Diff'],c='c',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['OWGR Diff'],c='k',alpha=0.7)
    plt.ylabel('OWGR 1 mon change'); plt.xticks(rotation=45)
    frame2 = plt.gca(); frame2.axes.xaxis.set_ticklabels([])

    plt.subplot(3,2,3)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['Temperature'],c='c',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['Temperature'],c='k',alpha=0.7)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['Humidity'],c='m',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['Humidity'],c='k',alpha=0.7)
    plt.ylabel('Temp & Humidity'); plt.xticks(rotation=45)
    frame3 = plt.gca(); frame3.axes.xaxis.set_ticklabels([])

    plt.subplot(3,2,4)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['Wind Speed'],c='b',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['Wind Speed'],c='k',alpha=0.7)
    plt.ylabel('Wind Speed'); plt.xticks(rotation=45)
    frame4 = plt.gca(); frame4.axes.xaxis.set_ticklabels([])
    # Gets rid of ticks and axis text: plt.xticks([])

    plt.subplot(3,2,5)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['Wind'],c='brown',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['Wind'],c='k',alpha=0.7)
    plt.xlabel('Date-Time'); plt.ylabel('Wind'); plt.xticks(rotation=45)

    plt.subplot(3,2,6)
    plt.plot(df_rolling_30m_avg.index, df_rolling_30m_avg['Score'],c='g',alpha=0.5)
    plt.plot(df_rolling_1hr_avg.index, df_rolling_1hr_avg['Score'],c='k',alpha=0.7)
    plt.xlabel('Date-Time'); plt.ylabel('Score Rel. To Par'); plt.xticks(rotation=45)
               
    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.2, hspace=0.25)

    fig = plt.gcf()
    fig.set_size_inches(11,7) # X,Y is width,heighth
    fig.savefig('2018 US Open - Training Set Rolling Averages - Weather and OWGR.png', bbox_inches='tight')
    
    plt.show()
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


    ''' Plot of each day's scores '''
    # I should have a round number column I can reference instead of the date
    # Or import the date_list
    df_rolling_R1_score_20m_avg = df.loc['2018-06-14'].rolling('1200s', min_periods=20).mean()
    df_rolling_R1_score_1hr_avg = df.loc['2018-06-14'].rolling('3600s', min_periods=60).mean()
    df_rolling_R2_score_20m_avg = df.loc['2018-06-15'].rolling('1200s', min_periods=20).mean()
    df_rolling_R2_score_1hr_avg = df.loc['2018-06-15'].rolling('3600s', min_periods=60).mean()
    df_rolling_R3_score_20m_avg = df.loc['2018-06-16'].rolling('1200s', min_periods=20).mean()
    df_rolling_R3_score_1hr_avg = df.loc['2018-06-16'].rolling('3600s', min_periods=60).mean()
    df_rolling_R4_score_20m_avg = df.loc['2018-06-17'].rolling('1200s', min_periods=20).mean()
    df_rolling_R4_score_1hr_avg = df.loc['2018-06-17'].rolling('3600s', min_periods=60).mean()
    
    plt.figure()
    
    plt.subplot(4,1,1)
    plt.plot(df_rolling_R1_score_20m_avg.index, df_rolling_R1_score_20m_avg['Score'], c='g', alpha=0.5)
    plt.plot(df_rolling_R1_score_1hr_avg.index, df_rolling_R1_score_1hr_avg['Score'], c='k', alpha=0.7)
    plt.xlabel(''); plt.ylabel('R1 Score'); plt.ylim(-0.3,0.7)
    frame1 = plt.gca(); frame1.axes.xaxis.set_ticklabels([])

    plt.subplot(4,1,2)
    plt.plot(df_rolling_R2_score_20m_avg.index, df_rolling_R2_score_20m_avg['Score'], c='g', alpha=0.5)
    plt.plot(df_rolling_R2_score_1hr_avg.index, df_rolling_R2_score_1hr_avg['Score'], c='k', alpha=0.7)
    plt.xlabel(''); plt.ylabel('R2 Score'); plt.ylim(-0.3,0.7)
    frame2 = plt.gca(); frame2.axes.xaxis.set_ticklabels([])

    plt.subplot(4,1,3)
    plt.plot(df_rolling_R3_score_20m_avg.index, df_rolling_R3_score_20m_avg['Score'], c='g', alpha=0.5)
    plt.plot(df_rolling_R3_score_1hr_avg.index, df_rolling_R3_score_1hr_avg['Score'], c='k', alpha=0.7)
    plt.xlabel(''); plt.ylabel('R3 Score'); plt.ylim(-0.3,0.7)
    frame3 = plt.gca(); frame3.axes.xaxis.set_ticklabels([])

    plt.subplot(4,1,4)
    plt.plot(df_rolling_R4_score_20m_avg.index, df_rolling_R4_score_20m_avg['Score'], c='g', alpha=0.5)
    plt.plot(df_rolling_R4_score_1hr_avg.index, df_rolling_R4_score_1hr_avg['Score'], c='k', alpha=0.7)
    plt.xlabel('DateTime'); plt.ylabel('R4 Score'); plt.xticks(rotation=45); plt.ylim(-0.3,0.7)

    fig = plt.gcf()
    fig.set_size_inches(6,5)
    fig.savefig('2018 US Open - Training Set Rolling Averages - Scores.png', bbox_inches='tight')
    
    plt.show()
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    
    return df












''' Machine Learning: Classifying every hole score relative to par '''
def MLHoleClassification(df_training_set):
    ''' EDA and Plotting: '''
    print('Training Set description:\n', df_training_set.describe())
    print(df_training_set.groupby('Hole').describe()) # Can use groupby to plot scores for each hole
    print(df_training_set.groupby('Hole').describe()['Score']) # Showing each hole's score stats
##    plt.figure()
##    scatter_matrix(df_training_set)
##    plt.show()

    ''' Replace scores with labels:
        -2.0 = '-2', -1.0 = '-1', 0.0 = '0', etc: '''
##    ''' Balancing the dataset: Converting all pars to 0, non-pars to 1 '''
##    df_training_set.Score = [0 if x==0 else 1 for x in df_training_set.Score]
    ''' Class counts: '''
    print('df_training_set Score counts:\n', df_training_set.Score.value_counts())

    ''' Checking for unbalanced dataset in non-scientific way: '''
##    df_score_counts = df_training_set.Score.value_counts()
##    df_score_counts_len = len(df_score_counts)
##    score_counts_sum = df_score_counts.sum()
##    for x in df_score_counts:
##        score_pct = x/score_counts_sum*100
##        # If scores are multiclass and one of them accounts for over 50% of the data:
##        if df_score_counts_len > 2 and score_pct > 50:
##            print('Warning: Possible unbalanced dataset:\n Class level',x,'is',score_pct,'% of dataset')

    ''' Checking for a large percentage of pars: '''
    df_score_par_pct = df_training_set.Score.value_counts(normalize=True).iloc[0]*100
    print('df_training_set.Score Par pct: %2f:' % df_score_par_pct)
    if df_score_par_pct > 50:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~Warning: Possible unbalanced dataset:')
        print('Pars comprise %.2f of df_training_set' % df_score_par_pct)



    ''' Resampling with replacement to bring minority scores up to the
        minimum number of n_splits required in StratifiedKFold() below: '''
    print('###############################  Dataset is being resampled')
    df_score_counts = df_training_set.Score.value_counts()
    score_count_maj_min_cutoff = 14
    upsample_to = 20
    df_score_counts_majority = df_score_counts[df_score_counts >= score_count_maj_min_cutoff]
    
    print('df_score_counts_majority:\n', df_score_counts_majority)
    df_score_counts_minority = df_score_counts[df_score_counts < score_count_maj_min_cutoff]
    print('df_score_scounts_minority:\n', df_score_counts_minority)

    ''' This loop grabs the minority score rows from dt_training_set that
        are below 5 samples and upsamples them. I specify the majority score
        in df_majority_scores and assume that it's always par. '''
    df_minority_upsampled = pd.DataFrame()
    minority_score_list = []
    for x in df_score_counts_minority.iteritems():
        minority_score, minority_score_count = x # x is a (score,count) tuple e.g. (4,3) or (5,1) or (6,1)
        minority_score_list.append(minority_score)
        # df_minority_scores contains all rows from df_training_set for which
        # the player got that minority score
        #print('Current minority score =', minority_score)
        df_minority = df_training_set[df_training_set.Score==minority_score]
        #print('df_minority before upsampling:\n', df_minority)
        df_minority_upsampled_unique = resample(df_minority,
                                         replace=True,
                                         n_samples=upsample_to,
                                         random_state=7)
        #print('Unique minority upsampled:\n', df_minority_upsampled_unique)
        df_minority_upsampled = df_minority_upsampled.append(df_minority_upsampled_unique)

    majority_score_list = []
    for x in df_score_counts_majority.iteritems():
          majority_score, majority_score_count = x
          majority_score_list.append(majority_score)

    df_majority = df_training_set[df_training_set.Score.isin(majority_score_list)]

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    print('Upsampled score counts:\n', df_upsampled.Score.value_counts())

    ''' Overwriting df_training_set for ML '''
    df_training_set = df_upsampled
    '''####################################'''

##    print('###############################  Removing all minority scores. This makes the previous upsampling pointless.\n')
##    df_training_set = df_majority
##    print('Majority score list:\n', majority_score_list)
##
##    print('df_training_set.head():\n', df_training_set.head())
##
##    print('###############################  Removing all players with OWGRs over 300\n')
##    df_training_set = df_training_set[df_training_set['New OWGR'] < 300]



    ''' One-hot encoding Hole numbers '''
    df = df_training_set.drop(['Player','DateTime','Hole','New OWGR','OWGR Diff','Temperature','Humidity'], axis=1)
    df_X = df.drop(['Score'], axis=1)

    # One-hot encode hole:
    if 'Hole' in df_X.columns:
        df_X = pd.get_dummies(df_X, columns=['Hole'], drop_first=True) # One-hot encoding Hole number
    if 'Pin' in df_X.columns:
        df_X = pd.get_dummies(df_X, columns=['Pin'], drop_first=True) # One-hot encoding Pin location

    df_y = df['Score']

    print('df head:\n', df.head())
    print('df_X head:\n', df_X.head())
    print('df_X.columns:\n', df_X.columns)
    print('df_y head:\n', df_y.head())

    X = df_X.values.astype(float)
    y = df_y.values.astype(int)
    y = y.astype(str)

    print('X[0:10]:\n', X[0:10])
    print('y[0:10]:\n', y[0:10])
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''


    # Split features from outputs
##    X = df_training_set.drop(['Player','DateTime','Score','Hole','New OWGR','OWGR Diff'], axis=1)
##    y = df_training_set.Score.astype('str')
##    print('Predicting: Hole Score')
##    print('X.head:\n', X.head())
##    print('y.head:\n', y.head())
    
    validation_size = 0.20
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

    print('y_train[0:20]:\n', y_train[0:20])
    df_y_train = pd.Series(y_train) # Turning them into dataframes 
    df_y_test = pd.Series(y_test)
    ''' Train and Test Set par counts: '''
    y_train_par_pct = df_y_train.value_counts(normalize=True).iloc[0]*100
##    y_train_par_pct = collections.Counter(y_train).iloc[0]*100
    y_test_par_pct = df_y_test.value_counts(normalize=True).iloc[0]*100
    y_test_par_pct_collections = collections.Counter(y_test)
    print('y_train Par pct: %.2f:' % y_train_par_pct)
    print('y_test Par pct: %.2f:' % y_test_par_pct)
    print('y_test_par_pct_collections:', y_test_par_pct_collections)

    
    scoring = 'neg_log_loss'

    # Spot Check Algorithms
    models = []
##    models.append(('LR', LogisticRegression()))
##    models.append(('LDA', LinearDiscriminantAnalysis()))
##    models.append(('KNN', KNeighborsClassifier()))
##    models.append(('CART', DecisionTreeClassifier()))
##    models.append(('NB', GaussianNB()))
##    models.append(('SVM Balanced', SVC(kernel='linear',
##                              class_weight='balanced'))) # Penalize
##    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier(n_estimators=100)))
    models.append(('GBC', GradientBoostingClassifier()))
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
##            kfold = model_selection.KFold(n_splits=10, random_state=seed)
##            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            stratifiedkfold = model_selection.StratifiedKFold(n_splits=5, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=stratifiedkfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    # Saving figure as PNG:
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    fig.savefig('2018 US Open - RF vs GBT - n_splits=10.png', bbox_inches='tight')
    plt.show()


##    # Predict with Logistic Regression:
##    lr = LogisticRegression()
##    lr.fit(X_train, y_train)
##    y_pred = lr.predict(X_train)
##    print('Logistic Regression Predictions:  ~~~~~~~~~~~~~~~~~~~~~~~~\n')
##    print('LR - Labels not predicted:', set(y_train) - set(y_pred))
##    print('LR - Accuracy:', accuracy_score(y_train, y_pred))
##    print('LR - Micro Precision:', precision_score(y_train, y_pred, average='micro'))
##    print('LR - Weighted Precision:', precision_score(y_train, y_pred, average='weighted'))
##    print('LR - Confusion Matrix:\n', confusion_matrix(y_train, y_pred))
##    print('LR - Classification Report:\n', classification_report(y_train, y_pred))
##    print('LR - Report using unique labels:  ~~~~~~~~~~~~~~~~~~~~~~\n')
##    print('LR - Unique F1 score:', metrics.f1_score(y_train, y_pred, average='weighted', labels=np.unique(y_pred)))
##    print('LR - Confusion Matrix with unique labels:\n', confusion_matrix(y_train, y_pred, labels=np.unique(y_pred)))
##    print('LR - Classification Report with unique labels:\n', classification_report(y_train, y_pred, labels=np.unique(y_pred)))

###############  SVM with balancing is taking hours to run.
##    # Predict with Random Forest:
##    svm = SVC() # penalize
##    svm.fit(X_train, y_train)
##    y_pred = svm.predict(X_test)
##    print('\nSupport Vector Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
##    print('SVM - Unique y_pred:', np.unique(y_pred))
##    print('SVM - Accuracy:', accuracy_score(y_test, y_pred))
##    print('SVM - Micro Precision:', precision_score(y_test, y_pred, average='micro'))
##    print('SVM - Weighted Precision:', precision_score(y_test, y_pred, average='weighted'))
    
    # Predict with Random Forest:
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)
    print('\nRandom Forest Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('RF - Unique y_pred:', np.unique(y_pred))
    print('RF - Accuracy:', accuracy_score(y_test, y_pred))
    print('RF - Micro Precision:', precision_score(y_test, y_pred, average='micro'))
    print('RF - Weighted Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('RF - y_pred_prob[0:2]:\n', y_pred_prob)
    print('RF - y_pred[0:2]:\n', y_pred)
    print('RF - Log Loss:', log_loss(y_test, y_pred_prob))
    

    # Predict with Gradient Boosting:
    gbt = GradientBoostingClassifier()
    gbt.fit(X_train, y_train)
    y_pred = gbt.predict(X_test)
    y_pred_prob = gbt.predict_proba(X_test)
    print('\nGradient Boosting Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('GBT - Unique y_pred:', np.unique(y_pred))
    print('GBT - Accuracy:', accuracy_score(y_test, y_pred))
    print('GBT - Micro Precision:', precision_score(y_test, y_pred, average='micro'))
    print('GBT - Weighted Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('GBT - y_pred_prob[0:10]:\n', y_pred_prob)
    print('GBT - Log Loss:', log_loss(y_test, y_pred_prob))

    ''' Plotting y_test vs y_pred '''
    y_test.astype(int); y_pred.astype(int)
    y_list = [[int(a),int(b)] for a,b in zip(y_test, y_pred)]
    #print('y_list[0:10]:\n', y_list[0:10])
    df_y = pd.DataFrame(y_list, columns=['y Test', 'y Pred'])
    df_y.sort_values(['y Test', 'y Pred'], inplace=True)
    #print('df_y.head():\n', df_y.head())
    y_test_sorted = df_y['y Test'].values.tolist()
    y_pred_sorted = df_y['y Pred'].values.tolist()
    
    plt.figure()
    plt.scatter(y_test_sorted, y_pred_sorted, marker='o', edgecolors='k', facecolors='none')
    plt.xlabel('Real Test Score'); plt.ylabel('Predicted Score')
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    fig.savefig('2018 US Open - ML Hole Classification - y_pred vs y_test.png', bbox_inches='tight')
    plt.show()
##    fig.suptitle('Predictions')
##    plt.plot(y_pred)

##    # Predictions and y_train info describing y_test data set
##    # and showing number of predictions for different scores:
##    print('\nPrediction counts:', pd.Series(y_pred).value_counts())
##    print('\nPrediction description:\n', pd.Series(y_pred).describe())
##    print('\ny_test:\n', y_test)
##    print('\ny_test description:\n', pd.Series(y_test).describe())
##    print('\ny_test counts:\n', pd.Series(y_test).value_counts())
##
    return results, X_test, y_test, y_pred, y_pred_prob








def MLRoundRegression(df2_training_set):
    ''' EDA and Plotting: '''
    print('Training Set 2 description:\n', df2_training_set.describe())
##    print(df2_training_set.groupby('Round').describe()) # Can use groupby to plot scores for each round
##    print(df2_training_set.groupby('Round').describe()['Score']) # Showing each round's score stats
##    plt.figure()
##    scatter_matrix(df2_training_set)
##    plt.show()

    ''' Replace scores with labels:
        -2.0 = '-2', -1.0 = '-1', 0.0 = '0', etc: '''
##    ''' Balancing the dataset: Converting all pars to 0, non-pars to 1 '''
##    df_training_set.Score = [0 if x==0 else 1 for x in df_training_set.Score]
    ''' Class counts: '''
    print('Score counts:\n', df2_training_set.Score.value_counts())

    ''' Checking for unbalanced dataset in non-scientific way: '''
    df_score_counts = df2_training_set.Score.value_counts()
    df_score_counts_len = len(df_score_counts)
    score_counts_sum = df_score_counts.sum()
    for x in df_score_counts:
        score_pct = x/score_counts_sum*100
        # If scores are multiclass and one of them accounts for over 50% of the data:
        if df_score_counts_len > 2 and score_pct > 50:
            print('Warning: Possible unbalanced dataset:\n Class level',x,'is',score_pct,'% of dataset')
    

##    ''' Resampling with replacement to bring minority scores up to the
##        minimum number of n_splits required in StratifiedKFold() below: '''
##    print('###############################  Dataset is being resampled')
##    score_count_maj_min_cutoff = 14
##    upsample_to = 20
##    df_score_counts_majority = df_score_counts[df_score_counts >= score_count_maj_min_cutoff]
##    
##    print('df_score_counts_majority:\n', df_score_counts_majority)
##    df_score_counts_minority = df_score_counts[df_score_counts < score_count_maj_min_cutoff]
##    print('df_score_scounts_minority:\n', df_score_counts_minority)
##
##    ''' This loop grabs the minority score rows from dt_training_set that
##        are below 5 samples and upsamples them. I specify the majority score
##        in df_majority_scores and assume that it's always par. '''
##    df_minority_upsampled = pd.DataFrame()
##    for x in df_score_counts_minority.iteritems():
##        minority_score, minority_score_count = x # x is a tuple e.g. (4,3) or (5,1) or (6,1)
##        # df_minority_scores contains all rows from df_training_set for which
##        # the player got that minority score
##        #print('Current minority score =', minority_score)
##        df_minority = df_training_set[df_training_set.Score==minority_score]
##        #print('df_minority before upsampling:\n', df_minority)
##        df_minority_upsampled_unique = resample(df_minority,
##                                         replace=True,
##                                         n_samples=upsample_to,
##                                         random_state=7)
##        #print('Unique minority upsampled:\n', df_minority_upsampled_unique)
##        df_minority_upsampled = df_minority_upsampled.append(df_minority_upsampled_unique)
##
##    majority_score_list = []
##    for x in df_score_counts_majority.iteritems():
##          majority_score, majority_score_count = x
##          majority_score_list.append(majority_score)
##
##    df_majority = df_training_set[df_training_set.Score.isin(majority_score_list)]
##
##    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
##    print('Upsampled score counts:\n', df_upsampled.Score.value_counts())
##
##    ''' Overwriting df_training_set for ML '''
##    df_training_set = df_upsampled
##    '''####################################'''

    df2_training_set.reset_index(inplace=True)
    # Split features from outputs
    print('###############################  Removing all players with OWGRs over 300\n')
    df2_training_set = df2_training_set[df2_training_set['New OWGR'] < 300] # Dropping all rows for which OWGR is greater than 300
    X = df2_training_set.drop(['Player','Tee Time','DateTime','DateTimeWeather','Round','Score'], axis=1)
    X['New OWGR'].astype('int')
    X['OWGR Diff'].astype('int')
    X.Temperature.astype('float')
    X.Humidity.astype('float')
    X['Wind Speed'].astype('float')
    X.Wind.astype('float')

    y = df2_training_set.Score.astype('int')

    print('X.head:\n', X.head())
    print('y.head:\n', y.head())

    validation_size = 0.20
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

    ''' Linear Regression '''
    scoring = 'r2'
    models = []
    models.append(('LR', LinearRegression()))
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            #stratifiedkfold = model_selection.StratifiedKFold(n_splits=5, random_state=seed)
            #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=stratifiedkfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    # Saving figure as PNG:
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    fig.savefig('2018 US Open - LinReg on Round data - n_splits=10.png', bbox_inches='tight')
    plt.show()

    
    # Predict with Random Forest:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('\nLinear Regression Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
##  Every lin reg prediction is unique, no need to look at this: print('LinReg - Unique y_pred:', np.unique(y_pred))
    print('LinReg - Variance score: %.2f' % metrics.r2_score(y_test, y_pred))
    print('LinReg - Mean Squ Error: %.2f' % metrics.mean_squared_error(y_test, y_pred))
    print('LinReg - Mean Abs Error: %.2f' % metrics.mean_absolute_error(y_test, y_pred))
    print('LinReg - Median Abs Err: %.2f' % metrics.median_absolute_error(y_test, y_pred))

    fig.suptitle('Predictions')

    plt.subplot(2,3,1)
    plt.scatter(X_test.Temperature, y_test, edgecolor='r', facecolor='none')
    plt.plot(X_test.Temperature, y_pred, color='k')
    plt.xlabel('Temperature'); plt.ylabel('Score')

    plt.subplot(2,3,2)
    plt.scatter(X_test.Humidity, y_test, edgecolor='orange', facecolor='none')
    plt.plot(X_test.Humidity, y_pred, color='k')
    plt.xlabel('Humidity'); plt.ylabel('Score')

    plt.subplot(2,3,3)
    plt.scatter(X_test['New OWGR'], y_test, marker='D', edgecolor='gray', facecolor='none')
    plt.plot(X_test, y_pred, color='k')
    plt.xlabel('New OWGR'); plt.ylabel('Score')

    plt.subplot(2,3,4)
    plt.scatter(X_test['OWGR Diff'], y_test, marker='D', edgecolor='gray', facecolor='none')
    plt.plot(X_test, y_pred, color='k')
    plt.xlabel('OWGR Diff'); plt.ylabel('Score')

    plt.subplot(2,3,5)
    plt.scatter(X_test['Wind Speed'], y_test, edgecolor='c', facecolor='none')
    plt.plot(X_test['Wind Speed'], y_pred, color='k')
    plt.xlabel('Wind Speed'); plt.ylabel('Score')

    plt.subplot(2,3,6)
    plt.scatter(X_test.Wind, y_test, edgecolor='m', facecolor='none')
    plt.plot(X_test.Wind, y_pred, color='k')
    plt.xlabel('Wind Dir'); plt.ylabel('Score')

    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.35, hspace=0.35)
    
    fig = plt.gcf()
    fig.set_size_inches(7.5,5)
    fig.savefig('2018 US Open - LinReg on Round data - y-predictions vs X-test', bbox_inches='tight')
    
    plt.show()
    
    return results, X_test, y_test, y_pred







def MLRoundClassification(df2_training_set):
    ''' EDA and Plotting: '''
    print('Training Set 2 description:\n', df2_training_set.describe())
##    print(df2_training_set.groupby('Round').describe()) # Can use groupby to plot scores for each round
##    print(df2_training_set.groupby('Round').describe()['Score']) # Showing each round's score stats
##    plt.figure()
##    scatter_matrix(df2_training_set)
##    plt.show()

    ''' Class counts: '''
    print('Score counts:\n', df2_training_set.Score.value_counts())

    ''' Checking for unbalanced dataset in non-scientific way: '''
    df_score_counts = df2_training_set.Score.value_counts()
    df_score_counts_len = len(df_score_counts)
    score_counts_sum = df_score_counts.sum()
    for x in df_score_counts:
        score_pct = x/score_counts_sum*100
        # If scores are multiclass and one of them accounts for over 50% of the data:
        if df_score_counts_len > 2 and score_pct > 50:
            print('Warning: Possible unbalanced dataset:\n Class level',x,'is',score_pct,'% of dataset')
    
    ''' Resampling with replacement to bring minority scores up to the
        minimum number of n_splits required in StratifiedKFold() below: '''
########  Resampling isn't working right now: ValueError: high <= low
##    print('###############################  Dataset is being resampled')
##    score_count_maj_min_cutoff = 2
##    upsample_to = 3
##    df_score_counts_majority = df_score_counts[df_score_counts >= score_count_maj_min_cutoff]
##    
##    print('df_score_counts_majority:\n', df_score_counts_majority)
##    df_score_counts_minority = df_score_counts[df_score_counts < score_count_maj_min_cutoff]
##    print('df_score_scounts_minority:\n', df_score_counts_minority)
##
##    ''' This loop grabs the minority score rows from dt_training_set that
##        are below 5 samples and upsamples them. I specify the majority score
##        in df_majority_scores and assume that it's always par. '''
##    df_minority_upsampled = pd.DataFrame()
##    for x in df_score_counts_minority.iteritems():
##        minority_score, minority_score_count = x # x is a tuple e.g. (4,3) or (5,1) or (6,1)
##        # df_minority_scores contains all rows from df_training_set for which
##        # the player got that minority score
##        #print('Current minority score =', minority_score)
##        df_minority = df_training_set[df_training_set.Score==minority_score]
##        #print('df_minority before upsampling:\n', df_minority)
##        df_minority_upsampled_unique = resample(df_minority,
##                                                replace=True,
##                                                n_samples=upsample_to,
##                                                random_state=7)
##        #print('Unique minority upsampled:\n', df_minority_upsampled_unique)
##        df_minority_upsampled = df_minority_upsampled.append(df_minority_upsampled_unique)
##
##    majority_score_list = []
##    for x in df_score_counts_majority.iteritems():
##          majority_score, majority_score_count = x
##          majority_score_list.append(majority_score)
##
##    df_majority = df2_training_set[df2_training_set.Score.isin(majority_score_list)]
##
##    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
##    print('Upsampled score counts:\n', df_upsampled.Score.value_counts())
##
##    ''' Overwriting df_training_set for ML '''
##    df2_training_set = df_upsampled
##    '''####################################'''


    df2_training_set.reset_index(inplace=True)
    print('df2_training_set.columns:\n', df2_training_set.columns)
    # Split features from outputs
    
    #print('###############################  Removing all players with OWGRs over 300\n')
    #df2_training_set = df2_training_set[df2_training_set['New OWGR'] < 300] # Dropping all rows for which OWGR is greater than 300

    df2_training_set['New OWGR'].astype('int')
    df2_training_set['OWGR Diff'].astype('int')
    df2_training_set.Temperature.astype('float')
    df2_training_set.Humidity.astype('float')
    df2_training_set['Wind Speed'].astype('float')
    df2_training_set.Wind.astype('float')
    
    ''' Round is being left out '''
    X = df2_training_set.drop(['Player','Tee Time','DateTime','DateTimeWeather','Round','New OWGR','OWGR Diff','Temperature','Humidity','Wind Speed','Speed','Firm','Score'], axis=1)

    y = df2_training_set.Score.astype('str')
    print('Score value counts:\n', y.value_counts(normalize=True))

    print('X.head:\n', X.head())
    print('y.head:\n', y.head())

    validation_size = 0.20
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

    ''' Classification: '''
    scoring = 'accuracy'
    models = []
##    models.append(('LogReg', LogisticRegress()))
##    models.append(('LDA', LinearDiscriminantAnalysis()))
##    models.append(('KNN', KNeighborsClassifier()))
##    models.append(('CART', DecisionTreeClassifier()))
##    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier(n_estimators=100)))
    models.append(('GBC', GradientBoostingClassifier()))
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
            kfold = model_selection.KFold(n_splits=5, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            #stratifiedkfold = model_selection.StratifiedKFold(n_splits=5, random_state=seed)
            #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=stratifiedkfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    # Saving figure as PNG:
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    fig.savefig('2018 US Open - Classification on Round data - n_splits=5.png', bbox_inches='tight')
    plt.show()

    # Predict with Random Forest:
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('\nRandom Forest Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('RF - Unique y_pred:', np.unique(y_pred))
    print('RF - Accuracy:', accuracy_score(y_test, y_pred))
    print('RF - Micro Precision:', precision_score(y_test, y_pred, average='micro'))
    print('RF - Weighted Precision:', precision_score(y_test, y_pred, average='weighted'))

    # Predict with Gradient Boosting:
    gbt = GradientBoostingClassifier()
    gbt.fit(X_train, y_train)
    y_pred = gbt.predict(X_test)
    print('\nGradient Boosting Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('GBT - Unique y_pred:', np.unique(y_pred))
    print('GBT - Accuracy:', accuracy_score(y_test, y_pred))
    print('GBT - Micro Precision:', precision_score(y_test, y_pred, average='micro'))
    print('GBT - Weighted Precision:', precision_score(y_test, y_pred, average='weighted'))

##    # Predict with Random Forest:
##    lr = LogisticRegression()
##    lr.fit(X_train, y_train)
##    y_pred = lr.predict(X_test)
##    print('\nLinear Regression Predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
####  Every lin reg prediction is unique, no need to look at this: print('LinReg - Unique y_pred:', np.unique(y_pred))
##    print('LinReg - Variance score: %.2f' % metrics.r2_score(y_test, y_pred))
##    print('LinReg - Mean Squ Error: %.2f' % metrics.mean_squared_error(y_test, y_pred))
##    print('LinReg - Mean Abs Error: %.2f' % metrics.mean_absolute_error(y_test, y_pred))
##    print('LinReg - Median Abs Err: %.2f' % metrics.median_absolute_error(y_test, y_pred))

##    fig.suptitle('Predictions')
##
##    plt.subplot(2,3,1)
##    plt.scatter(X_test.Temperature, y_test, edgecolor='r', facecolor='none')
##    plt.plot(X_test.Temperature, y_pred, color='k')
##    plt.xlabel('Temperature'); plt.ylabel('Score')
##
##    plt.subplot(2,3,2)
##    plt.scatter(X_test.Humidity, y_test, edgecolor='orange', facecolor='none')
##    plt.plot(X_test.Humidity, y_pred, color='k')
##    plt.xlabel('Humidity'); plt.ylabel('Score')
##
##    plt.subplot(2,3,3)
##    plt.scatter(X_test['New OWGR'], y_test, marker='D', edgecolor='gray', facecolor='none')
##    plt.plot(X_test, y_pred, color='k')
##    plt.xlabel('OWGR'); plt.ylabel('Score')
##
##    plt.subplot(2,3,4)
##    plt.scatter(X_test['OWGR Diff'], y_test, marker='D', edgecolor='gray', facecolor='none')
##    plt.plot(X_test, y_pred, color='k')
##    plt.xlabel('OWGR Diff'); plt.ylabel('Score')
##
##    plt.subplot(2,3,5)
##    plt.scatter(X_test['Wind Speed'], y_test, edgecolor='c', facecolor='none')
##    plt.plot(X_test['Wind Speed'], y_pred, color='k')
##    plt.xlabel('Wind Speed'); plt.ylabel('Score')
##
##    plt.subplot(2,3,6)
##    plt.scatter(X_test.Wind, y_test, edgecolor='m', facecolor='none')
##    plt.plot(X_test.Wind, y_pred, color='k')
##    plt.xlabel('Wind Dir'); plt.ylabel('Score')
##
##    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.35, hspace=0.35)
##    
##    fig = plt.gcf()
##    fig.set_size_inches(7.5,5)
##    fig.savefig('2018 US Open - LinReg on Round data - y-predictions vs X-test', bbox_inches='tight')
##    
##    plt.show()

    ''' Plotting y_test vs y_pred '''
    y_test.astype(int); y_pred.astype(int)
    y_list = [[int(a),int(b)] for a,b in zip(y_test, y_pred)]
    print('y_list[0:10]:\n', y_list[0:10])
    df_y = pd.DataFrame(y_list, columns=['y Test', 'y Pred'])
    df_y.sort_values(['y Test', 'y Pred'], inplace=True)
    print('df_y.head():\n', df_y.head())
    y_test_sorted = df_y['y Test'].values.tolist()
    y_pred_sorted = df_y['y Pred'].values.tolist()
    
    plt.figure()
    plt.scatter(y_test_sorted, y_pred_sorted, marker='o', edgecolors='k', facecolors='none')
    plt.xlabel('Real Test Score'); plt.ylabel('Predicted Score')
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    fig.savefig('2018 US Open - ML Round Classification - y_pred vs y_test.png', bbox_inches='tight')
    plt.show()

    
    return results, X_test, y_test, y_pred





def KerasNN(df_training_set):
    seed = 7
    np.random.seed(seed)

    df = df_training_set.drop(['Player','DateTime','New OWGR','OWGR Diff','Temperature','Speed','Firm'], axis=1)
    # Old way of getting the dummy variables before 'drop_first' was introduced:
    #hole_dummies = pd.get_dummies(df.Hole, prefix='Hole').iloc[:,1:]
    #print('hole_dummies:\n', hole_dummies)
    #df = pd.concat([df,hole_dummies], axis=1)
    #df.drop(columns=['Hole'], inplace=True)
    df_X = df.drop(['Score'], axis=1)

    # One-hot encode hole:
    if 'Hole' in df_X.columns:
        df_X = pd.get_dummies(df_X, columns=['Hole'], drop_first=True)
    if 'Pin' in df_X.columns:
        df_X = pd.get_dummies(df_X, columns=['Pin'], drop_first=True)
    df_y = df['Score']
    print('df head:\n', df.head())
    print('df_X head:\n', df_X.head())
    print('df_y head:\n', df_y.head())

    X = df_X.values.astype(float)
    y = df_y.values.astype(int)
    y = y.astype(str)

    print('X[0:10]:\n', X[0:10])
    print('y[0:10]:\n', y[0:10])

    ''' Encode class values as integers '''
    le = LabelEncoder()
    le.fit(y) # fit() figures out which classes exist in y, maps them onto the label encoder le.
    print('le.classes_:', le.classes_) # The classes 'vertosa', etc are stored in le.classes_
    encoded_y = le.transform(y) # Transforms classes ('vertosa', etc) into integers 0, 1, 2, etc
    print('encoded_y:\n', encoded_y)
    ''' Convert integers to dummy variables (i.e. one hot encoded) '''
    dummy_y = np_utils.to_categorical(encoded_y) # Converts integers to categorical e.g. one-hot matrix [[1,0,0],...,[0,1,0],...,[0,0,1]]
    print('dummy_y:\n', dummy_y)


    def baseline_model():
        # Create model:
        model = Sequential()
        num_inputs = len(df_X.columns)
        print('num_inputs:', num_inputs)
        model.add(Dense(26, input_dim=num_inputs, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        # Compile model:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=72, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    ''' Evaluating our model (estimator) on the dataset (X and dummy_y) using
        a 10-fold cross-validation procedure (kfold): '''
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print('Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
##    ce = keras.losses.categorical_crossentropy(y_true, y_pred)
##    print('Crossentropy: %.2f' % ce)


    return results







'''#########################################################################################################################'''
# Returns df of yearly TPC stadium course length, tour driving dist (not Players
# field driving dist), and other yearly variables
def PlayersYearly():
    global year_start, year_end
    year_list = []
    for year in range(year_start,year_end+1):
        year_list.append(year)
    year_series = pd.Series(year_list)

    print('year list:\n', year_list)
    # Drive distance values:
    # https://golfweek.com/2015/12/22/average-driving-distance-pga-tour-hasnt-changed-much-decade/
    # https://www.pgatour.com/content/dam/pgatour/shotlink/rutgers.pdf
    avg_driving_dist = np.array([257, 257, 257, 257, 257, 257, 257, 260,
                        262.5, 261, 260.2, 260, 262, 263.5, 266, 267.5, 270.1, 272.7,
                        273, 279.2, 279.9, 287.2, 288, 288.9, 289.8, 288.6, 287.3, 287.4,
                        287.3, 290.9, 290, 289.1, 287.2, 288.8, 289.7, 290, 296])
    avg_driving_dist.reshape(len(avg_driving_dist),-1)

    course_length = np.array([6857, 6857, 6857, 6857, 6857, 6857, 6857, 6857,
                     6896, 6896, 6896, 6896, 6896, 6896, 6896, 6896, 6950, 7093,
                     7093, 7093, 7093, 7093, 7093, 7093, 7098, 7215, 7215, 7215,
                     7215, 7215, 7215, 7215, 7215, 7215, 7215, 7189, 7189])
    course_length.reshape(len(course_length),-1)
    
    dist_length_ratio = avg_driving_dist/course_length*100 # Avg driving dist to course length ratio
    
    total_dist_length_ratio = (avg_driving_dist*14)/course_length*100 # Total driving dist over 14 holes to length ratio
    
    avg_driving_dist.tolist()
    course_length.tolist()
    dist_length_ratio.tolist()
    total_dist_length_ratio.tolist()

    # Combining all arrays into one. Method np.column_stack() is crucial to doing this.
    # Above I turn all np arrays into lists, do I need to do this? ****************
    yearly_stats = np.column_stack([avg_driving_dist, course_length, dist_length_ratio, total_dist_length_ratio])
    print('yearly stats:\n', yearly_stats)
    col_names = ['Avg Driving Dist.','Course Length','Ratio','Total Dist Ratio']

    df = pd.DataFrame(yearly_stats, columns=col_names)

    df['Year'] = year_series.values
    df.set_index('Year', inplace=True)
    print('df: Players yearly data:\n', df)
    return df

# Scrapes official world golf rankings by the week, returns df for specified years and weeks.
#def owgr():
    # Go to website: http://www.owgr.com/about?tabID={BBE32113-EBCB-4AD1-82AA-E3FE9741E2D9}
    # Click on dropdown to select year or...
    # Step through yearly websites by creating the yearly link:
    # http://www.owgr.com/about?tabID={BBE32113-EBCB-4AD1-82AA-E3FE9741E2D9}&year=2011


#####################  Plotting  #######################
# Plot score data
def PlotPlayersWinningScores(df):
    global year_start, year_end, num_years
##    df = df.reindex(index=df.index[::-1], inplace=True)
##    print('df:\n', df)
    # Flipping the index so the plot starts with oldest year e.g. 1982
    df = df.iloc[::-1]
    ##print('df:\n', df)

    plt.figure()
    plt.suptitle('Winning Scores Stats - 1982-2018')

    # Winning scores over time:
    plt.subplot(121)
    plt.scatter(df.index, df['Winning Score'], s=4, c='k'); plt.xlabel('Year'); plt.ylabel('Winning Score')
    plt.xticks(list(range(0,year_end+1-year_start,8))) # For some reason it won't take range(year_start,year_end+1,2)
    # Hist:
    plt.subplot(122)
    plt.hist(df['Winning Score'], bins=14, color='#c9d5ff', edgecolor='k')
    plt.xlabel('Winning Score'); plt.ylabel('Count');

    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.28)
    
    # Saving figure as PNG:
    fig = plt.gcf()
    fig.set_size_inches(6,3)
    fig.savefig('Players - Winning Scores - 1982-2018.png', bbox_inches='tight')

    plt.show()

def PlotPlayersDailyScores(df):
    print('df:\n', df)
    global year_start, year_end, num_years
    
    data_list = df.values.tolist()
    ##print('data list:\n', data_list)

    total_sunday = []; total_saturday = []; total_friday = []; total_thursday = []

    # For loop jumps every 4 yrs to grab every Sunday list because
    # data is listed as [[sun],[sat],[fri],[thurs],[sun],...]
    for i in range(0,num_years*4,4):
        # Sunday:
        sunday = data_list[i][2]; sunday = list(filter(None, sunday)); sunday = [int(x) for x in sunday]
        total_sunday.extend(sunday)
        ##print('total_sunday:\n', total_sunday)        
        # Saturday:
        saturday = data_list[i+1][2]; saturday = list(filter(None, saturday)); saturday = [int(x) for x in saturday]
        total_saturday.extend(saturday)
        # Friday:
        friday = data_list[i+2][2]; friday = list(filter(None, friday)); friday = [int(x) for x in friday]
        total_friday.extend(friday)
        # Thursday:
        thursday = data_list[i+3][2]; thursday = list(filter(None, thursday)); thursday = [int(x) for x in thursday]
        total_thursday.extend(thursday)

    df_sunday = pd.DataFrame(total_sunday, columns=['daily scores'])
    ##print('df_sunday:\n', df_sunday)
    df_saturday = pd.DataFrame(total_saturday, columns=['daily scores'])
    df_friday = pd.DataFrame(total_friday, columns=['daily scores'])
    df_thursday = pd.DataFrame(total_thursday, columns=['daily scores'])

    # Collecting each year's daily scores into their own lists for comparing each
    # year's average and median scores
    # This next loop is ugly because data_list is organized with year and date
    # as indeces.
    total_day_scores = []; total_year = []
    day_scores = []; current_year = []
    year_count = 0
    year_list_str = list(range(year_end,year_start-1,-1)) # From 2018 to non-inclusive 1981
    year_list = [int(x) for x in year_list_str]
    #print('year list:\n', year_list)
    for i in range(1,num_years*4):
        day_scores.extend(data_list[i][2])
        if i%4 == 0:
            day_scores = list(filter(None, day_scores)) # Remove ''
            day_scores = [int(x) for x in day_scores]   # Convert all str elements to ints
            current_year = [year_list[year_count]]*len(day_scores)  # Make year list as long as day_scores list
            #print('year_list:\n', year_list)
            # Compile all day_scores:
            total_day_scores.extend(day_scores)
            total_year.extend(current_year)
            day_scores = []; current_year = []
            year_count += 1

##    print('total_day_scores:\n', total_day_scores)
##    print('total_year:\n', total_year)
##    print('length of total_day_scores:\n', total_day_scores)
##    print('length of total_year:\n', total_year)
    total = [total_year,total_day_scores]
    total = list(map(list, zip(*total))) # Transposing the list
    #print('total:\n', total)

    df_yearly_stats = pd.DataFrame(total, columns=['Year', 'Daily Scores'])
    #print('df_yearly_stats:\n', df_yearly_stats)


    # Pull Daily Scores:
    # Calculate length of first 4 Daily Scores rows, make year list of same length
    # Repeat every 4th row, extending each of the two lists
    # year_daily_score_list = total_daily_scores.append(year_list)
    # df_yearly_stats = pd.DataFrame(year_daily_score_list, columns=['Year', 'Daily Scores'])
    
    # Boxplot of each year's field's scoring statistics (mean, median, stddev):
    # Using Pandas boxplot function
    plt.figure()
    df_yearly_stats.boxplot(column=['Daily Scores'], by='Year', grid=False)
    plt.xlabel('Year'); plt.ylabel('Score')
    plt.xticks(rotation=90)

    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(9,5)
    fig.savefig('Players - Yearly Score Distributions Boxplot - 1982-2018.png', bbox_inches='tight')
    
    plt.show()

    # Daily distributions for all years:
    plt.suptitle('Daily Score Distributions - 1982-2018')

    bin_range = range(59,91,1)
    density_max = 0.15
    # Specifying x-ticks at every integer: plt.xticks(list(range(59,91,1)))
    # Sunday:
    plt.subplot(221)
    ##print('df_sunday daily scores:\n', df_sunday['daily scores'])
    plt.hist(df_sunday['daily scores'], density=1, bins=bin_range, color='#ff851c', edgecolor='k')
    plt.axis([59,91,0,density_max]); plt.xlabel('Sunday Scores'); plt.ylabel('Probability')
    # Saturday:
    plt.subplot(222)
    plt.hist(df_saturday['daily scores'], density=1, bins=bin_range, color='#ff851c', edgecolor='k')
    plt.axis([59,91,0,density_max]); plt.xlabel('Saturday Scores')
    # Friday:
    plt.subplot(223)
    plt.hist(df_friday['daily scores'], density=1, bins=bin_range, color='#ff851c', edgecolor='k')
    plt.axis([59,91,0,density_max]); plt.xlabel('Friday Scores'); plt.ylabel('Probability'); 
    # Thursday:
    plt.subplot(224)
    plt.hist(df_thursday['daily scores'], density=1, bins=bin_range, color='#ff851c', edgecolor='k')
    plt.axis([59,91,0,density_max]); plt.xlabel('Thursday Scores')

    # Adjust subplots size:
    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.3, hspace=0.32)
    
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    fig.savefig('Players - Daily Scores - 1982-2018.png', bbox_inches='tight')

    plt.show()
    return df_yearly_stats


def PlotPlayersYearly(df):
    plt.figure(1)
    plt.suptitle('Yearly Stats - 1982-2018')

    # Avg Driving Distance:
    plt.subplot(1,3,1)
    plt.scatter(df.index, df['Avg Driving Dist.'], s=4, c='#82c8ff')
    plt.xlabel('Year'); plt.ylabel('Avg Driving Distance, yards')
    # Sawgrass Course Length:
    plt.subplot(1,3,2)
    plt.scatter(df.index, df['Course Length'], s=4, c='m')
    plt.xlabel('Year'); plt.ylabel('Course Length, yards')
    # Total Driving Distance / Course Length ratio:
    plt.subplot(1,3,3)
    plt.scatter(df.index, df['Total Dist Ratio'], s=4, c='#a51212')
    plt.xlabel('Year'); plt.ylabel('Total D/L, %')

    # Adjust subplots size:
    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.42)
    
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(9,3)
    fig.savefig('Players - Yearly Stats - 1982-2018.png', bbox_inches='tight')

    plt.show()


# Plot weather data for a df containing a single day. Need to update this to handle the
# multi-day weather df
def PlotSingleDayWeather(df, year, date):
    df = df.loc[year,date] # Select only a portion of processed data
    print('df for date,',date,':\n', df)
    print('df index:\n', df.index)
    df.reset_index(inplace=True)
    print('df no index:\n', df)
    df.fillna(0, inplace=True)
    
    #time_list = [t.isoformat() for t in df.Time] # isoformat(): '10:00:00'
    #print('time_list =', time_list)
    hour_list = [t.hour for t in df.Time]
    print('hour_list =', hour_list)

    plt.figure()
    plt.subplot(321)
    plt.plot(hour_list, df['Temperature'], color='k'); plt.xlabel('Time, hours'); plt.ylabel('Temp, F')
    plt.subplot(322)
    plt.plot(hour_list, df['Humidity'], color='r'); plt.xlabel('Time, hours'); plt.ylabel('Humidity, %')
    plt.subplot(323)
    plt.plot(hour_list, df['Wind Speed'], color='y'); plt.xlabel('Time, hours'); plt.ylabel('Wind Speed, mph')
    plt.subplot(324)
    plt.plot(hour_list, df['Wind Gust'], color='c'); plt.xlabel('Time, hours'); plt.ylabel('Wind Gust, mph') 
    plt.subplot(325)
    plt.plot(hour_list, df['Precip.'], color='g'); plt.xlabel('Time, hours'); plt.ylabel('Precipitation, in')
    plt.subplot(326)
    plt.plot(hour_list, df['Pressure'], color='orange'); plt.xlabel('Time, hours'); plt.ylabel('Pressure, in')

    # Adjust subplots size:
    plt.subplots_adjust(left=0.12, bottom=0.18, wspace=0.4)

    plt.suptitle('Single Day Weather - '+date)

    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    fig.savefig('Players - Single Day Weather - '+date+'.png', bbox_inches='tight')

    plt.show()

def PlotMultiDayWeather(df):
    df.reset_index(inplace=True) # Getting rid of indices so plotting works. Not sure why indices are a problem.
    print('df: indices have been removed for plotting:\n', df)

##    plt.figure()
##    plt.subplot(1,4,1)
##    plt.plot(df.Date, df['Temperature'], color='k'); plt.xlabel('Date'); plt.ylabel('Temp, F')
##    plt.subplot(1,4,2)
##    plt.plot(df.Date, df['Humidity'], color='r'); plt.xlabel('Date'); plt.ylabel('Humidity, %')
##    plt.subplot(1,4,3)
##    plt.scatter(df.Date, df['Precip.'], s=10, alpha=0.5, facecolors='none', edgecolors='g'); plt.xlabel('Date'); plt.ylabel('Precipitation, in')
##    plt.subplot(1,4,4)
##    plt.scatter(df.Date, df['Wind Gust'], s=10, alpha=0.5, facecolors='none', edgecolors='b'); plt.xlabel('Date'); plt.ylabel('Wind Gust, mph')
##    plt.show()

    # Box plots:

##    plot = fig.add_subplot(111)
##    plot.tick_params(axis='bottom', which='major', labelsize=8)
##    plot.tick_params(axis='bottom', which='minor', labelsize=8)
##    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True) # Creating subplots. THIS DOESN'T WORK
    
    boxplot_by_str = 'Date' # Organize by 'Date' or 'Year' or 'Time'
    figsize_x = 14.5
    figsize_y = 4
    figsize=(figsize_x, figsize_y)

    fig = plt.figure(1)

    # Would like to separate x-axis labels by year despite data organized by date
    # Temperature box plot by year
    df.boxplot(column=['Temperature'], by=boxplot_by_str, grid=False, figsize=figsize)
    plt.xlabel(boxplot_by_str); plt.ylabel('Temp, F'); plt.tick_params(axis='x', which='major', labelsize=6)
    plt.xticks(rotation=90)
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(figsize_x, figsize_y)
    fig.savefig('Players - Multiday Weather - Temperature.png', bbox_inches='tight')
    
    # Humidity
    bp = df.boxplot(column=['Humidity'], by=boxplot_by_str, grid=False, figsize=figsize)
    plt.xlabel(boxplot_by_str); plt.ylabel('Humidity, %'); plt.tick_params(axis='x', which='major', labelsize=6)
    plt.xticks(rotation=90)
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(figsize_x, figsize_y)
    fig.savefig('Players - Multiday Weather - Humidity.png', bbox_inches='tight')

    # Precipitation box plot by year
    bp = df.boxplot(column=['Precip.'], by=boxplot_by_str, grid=False, figsize=figsize)
    plt.xlabel(boxplot_by_str); plt.ylabel('Precipitation, in'); plt.tick_params(axis='x', which='major', labelsize=6)
    plt.xticks(rotation=90)
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(figsize_x, figsize_y)
    fig.savefig('Players - Multiday Weather - Precipitation.png', bbox_inches='tight')

    # Wind Speed
    bp = df.boxplot(column=['Wind Speed'], by=boxplot_by_str, grid=False, figsize=figsize)
    plt.xlabel(boxplot_by_str); plt.ylabel('Wind Speed'); plt.tick_params(axis='x', which='major', labelsize=6)
    plt.xticks(rotation=90)
    # Save figure:
    fig = plt.gcf()
    fig.set_size_inches(figsize_x, figsize_y)
    fig.savefig('Players - Multiday Weather - Wind Speed.png', bbox_inches='tight')

    plt.show()
    
# Pickle in:
def PickleInPlayersScores():
    df = pd.read_pickle('players_winning_scores.pickle')
    df.rename(index=str, columns={'year':'Year', 'winning_score':'Winning Score'}, inplace=True)
    print('df: Players winning scores:\n', df)
    return df

def PickleInPlayersDailyScores():
    df = pd.read_pickle('players_daily_scores.pickle')
    #df.set_index('Year', inplace=True)
    print('df: Players daily scores:\n', df)
    return df

# Pickle in:
def PickleInYearDateLists(year_pickle_file, date_pickle_file):
    with open(year_pickle_file, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pickle_file, 'rb') as dp:
        date_list = pickle.load(dp)

    return year_list, date_list

def PickleInWeather(weather_pickle_file, year_pickle_file, date_pickle_file):
    df = pd.read_pickle(weather_pickle_file)

    with open(year_pickle_file, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pickle_file, 'rb') as dp:
        date_list = pickle.load(dp)

    print('df: Unprocessed Weather:\n', df)
    print('Year list:\n', year_list)
    print('Date list:\n', date_list)
    
    return df, year_list, date_list

def PickleInWeatherProcessed(weather_proc_pickle_file, year_pickle_file, date_pickle_file):
    df_proc_weather = pd.read_pickle(weather_proc_pickle_file)

    with open(year_pickle_file, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pickle_file, 'rb') as dp:
        date_list = pickle.load(dp)

    print('df: Processed Weather:\n', df_proc_weather)
    print('Year list:\n', year_list)
    print('Date list:\n', date_list)
    return df_proc_weather, year_list, date_list


def PickleInWeatherProcessedImputed(weather_proc_imp_pickle_file):
    df_proc_imp_weather = pd.read_pickle(weather_proc_imp_pickle_file)

    with open(year_pickle_file, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pickle_file, 'rb') as dp:
        date_list = pickle.load(dp)

    print('df: Processed Imputed Weather:\n', df_proc_imp_weather)
    return df_proc_imp_weather, year_list, date_list


def PickleInPlayerList(player_pickle_file): # Works after GetTeeTimes has generated and cleaned a player list
    with open(player_pickle_file, 'rb') as plp:
        player_list = pickle.load(plp)

    print('player_list:\n', player_list)
    return player_list

def PickleInScorecards(scorecards_pkl):
    df_scorecards = pd.read_pickle(scorecards_pkl)
    print('df: US Open ESPN scorecards:\n', df_scorecards)
    return df_scorecards
                                                          

def PickleIn_SC_TT_OWGR(scorecards_pkl, teetimes_pkl, owgr_pkl, year_pkl, date_pkl):
    df_scorecards = pd.read_pickle(scorecards_pkl)
    print('df: US Open ESPN scorecards:\n', df_scorecards)
    df_tee_times = pd.read_pickle(teetimes_pkl)
    print('df: US Open Tee Times:\n', df_tee_times)
    df_owgr = pd.read_pickle(owgr_pkl)
    print('df: US Open OWGR:\n', df_owgr)
    
    with open(year_pkl, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pkl, 'rb') as dp:
        date_list = pickle.load(dp)
    
    return df_scorecards, df_tee_times, df_owgr, year_list, date_list


def PickleInRoundProgress(round_progress_pkl, round_progress_final_score_pkl):
    df_round_progress = pd.read_pickle(round_progress_pkl)
    print('df: US Open Round Progress:\n', df_round_progress)

    df_round_progress_final_score = pd.read_pickle(round_progress_final_score_pkl)
    print('df: US Open Round Progress Final Score:\n', df_round_progress_final_score)

    return df_round_progress, df_round_progress_final_score


def PickleInProcImpWeatherAndRoundProgress(round_progress_pkl, round_progress_final_score_pkl, weather_proc_imp_pkl, year_pkl, date_pkl):
    df_round_progress = pd.read_pickle(round_progress_pkl)
    df_round_progress_final_score = pd.read_pickle(round_progress_final_score_pkl)
    df_proc_imp_weather = pd.read_pickle(weather_proc_imp_pkl)
    
    print('df: US Open Round Progress:\n', df_round_progress)
    print('df: US Open Round Progress Final Score:\n', df_round_progress_final_score)
    print('df: US Open Processed Imputed Weather:\n', df_proc_imp_weather)

    with open(year_pkl, 'rb') as yp:
        year_list = pickle.load(yp)
    with open(date_pkl, 'rb') as dp:
        date_list = pickle.load(dp)
    
    return df_round_progress, df_round_progress_final_score, df_proc_imp_weather, year_list, date_list

def PickleInTrainingSet(training_set_pkl, training_set_2_pkl):
    df_training_set = pd.read_pickle(training_set_pkl)
    df2_training_set = pd.read_pickle(training_set_2_pkl)

    print('df_training_set:\n', df_training_set)
    print('df2_training_set:\n', df2_training_set)

    return df_training_set, df2_training_set

##################  Running Functions  ###########################################

#### WINNING AND DAILY SCORES: Full range is 1982 to 2018
#players_winning_scores, players_daily_scores = GetPlayersData()
#df_players_winning_scores = PickleInPlayersScores()
#PlotPlayersWinningScores(df_players_winning_scores)


####  WEATHER:
''' Get Weather from webpage '''
#df, year_list, date_list = GetWeatherData('2018_us_open_dates.csv','KFOF') # Pickles out the df, year and date lists
''' Process Weather: Removing units from data strings and converting to integers and floats '''
#df, year_list, date_list = PickleInWeather('weather.pickle','year_list.pkl','date_list.pkl') # Specify full pickle file names
#df_proc_weather, total_sub_list = ProcessWeatherData(df,year_list,date_list) # Pickles out df, no need to pickle out year and date list
#df_merged = MergeTwoWeathers('players_weather_1986-2018_missing_2013-5-11.pickle', 'players_weather_2013-5-11.pickle')
''' Impute Weather to the minute aka subhourly: '''
#df_proc_weather, year_list, date_list = PickleInWeatherProcessed('weather_processed.pickle','year_list.pkl','date_list.pkl')
#df_proc_imp_weather = ImputeWeatherData(df_proc_weather, year_list, date_list) # Might also return: master_ffit_noise_temperature, master_ffit_noise_humidity, master_ffit_noise_wind_speed 
''' Plotting Weather: '''
#PlotSingleDayWeather(df_proc_weather,year,date) # year = '2018', day = '2018-06-14'. This will change to day only
#PlotMultiDayWeather(df_proc_weather)
#df_proc_imp_weather, year_list, date_list = PickleInWeatherProcessedImputed('weather_processed_imputed.pickle')


####  TEE TIMES:
#year_list, date_list = PickleInYearDateLists('year_list.pkl','date_list.pkl')
#df_tee_times, player_list = GetTeeTimes(year_list, date_list)

####  PLAYER STATS:
player_list = PickleInPlayerList('player_list.pkl')
#df = GetPlayerStatsESPN(player_list, '2018_us_open_player_stats_espn.csv')
df = GetPlayerStatsPGATourWebsite(player_list, '2018_player_stats_pga.csv','2018_player_stats_euro.csv')

####  OWGR:
''' Imports CSV of a particular week's OWGR (csv file) for all of the world's players, converts to tournament
    specific OWGR and pickles out both tournament and world OWGR dataframes. Tournament specific OWGR dataframe
    can be converted to a list and multipled by 18 to be inserted into df_round_progress in GetRoundProgress. '''
''' Note: GetOWGR requires a tournament specific tee times player list, and is therefore run after GetTeeTimes '''
#player_list = PickleInPlayerList('player_list.pkl')
#df_owgr_new = GetOWGR(player_list, '2018_us_open_owgr_wk23.csv', '2018_us_open_owgr_wk19.csv') # Provide owgr csv file name


####  SCORECARDS:
''' Scrapes ESPN's player profile webpages for scores. Missing many players. Fix by scraping leaderboard
    webpage for tournament of interest.'''
#df_scorecards, df_player_name_ids, master_score_list, master_hole_list, master_yardage_list, master_par_list = GetScorecardsESPN()
''' Scorecards EDA. Requires Training Set becuase there are no hole numbers present, I need to build these in. '''
#df_training_set = PickleInTrainingSet('us_open_training_set.pickle')
#df_training_set = ScorecardsEDA(df_training_set)

####  ROUND PROGRESSION:
''' First pickle in Scorecards (SC), Tee Times (TT) and Tournament specific OWGR (OWGR). Once
    these are in, OWGR is inserted into df_round_progress in preparation for running the data
    through AI()'''
''' Why am I returning df_scorecards and df_tee_times in GetRoundProgress? I don't believe these
    dataframes are being modified here. '''
#df_scorecards, df_tee_times, df_owgr, year_list, date_list = PickleIn_SC_TT_OWGR('us_open_scorecards.pickle','us_open_tee_times.pickle','us_open_owgr.pickle','year_list.pkl','date_list.pkl')
#df_round_progress, df_round_progress_final_score, df_scorecards, df_tee_times, df_owgr = GetRoundProgress(df_scorecards, df_tee_times, df_owgr, year_list, date_list) # df_scorecards and df_tee_times are modified in GetRoundProgress

####  AI:
''' Build the training set from Round Progress and Weather: '''
#df_round_progress, df_round_progress_final_score, df_proc_imp_weather, year_list, date_list = PickleInProcImpWeatherAndRoundProgress('us_open_round_progress.pickle','us_open_round_progress_final_score.pickle','weather_processed_imputed.pickle','year_list.pkl','date_list.pkl')
#df_training_set, df2_training_set = MakeTrainingSet(df_proc_imp_weather, df_round_progress, df_round_progress_final_score, year_list, date_list)
#dfs = RoundProgressEDA(df_round_progress)
''' Machine learning: '''
#df_training_set, df2_training_set = PickleInTrainingSet('us_open_training_set.pickle','us_open_training_set_2.pickle')
#df = TrainingSetEDA(df_training_set) # Plots Temp, Wind Speed, OWGR, etc vs Date-Time
#results, X_test, y_test, y_pred, y_pred_prob = MLHoleClassification(df_training_set)
#results, X_test, y_test, y_pred = MLRoundRegression(df2_training_set)
#results, X_test, y_test, y_pred = MLRoundClassification(df2_training_set)

''' Keras Deep Learning '''
#results = KerasNN(df_training_set)

#### YEARLY PLAYERS DATA:
#df_players_yearly = PlayersYearly() # df_players_yearly contains driving dist and course length for each year
#df_yearly_stats = PlotPlayersYearly(df_players_yearly)
#df_players_daily_scores = PickleInPlayersDailyScores()
#PlotPlayersDailyScores(df_players_daily_scores)

###################################################################################
