import datetime as dt
import logging
import re

import numpy as np

from global_functions import retrieve_content, set_logging_Config, save_data_to_hdf5


def get_detailed_and_week_table(tables):
    """ identify the relevant tables to retrieve the forecasts for today, tomorrow and next week

    :param tables: the list of tables from which the relevant tables are to be identified (list of pandas DataFrames)
    :return: the detailed table containing the forecasts for today and tomorrow and the week table containing the
     forecast for next week (list of pandas DataFrames)
    """
    detailed, week = None, None
    for table in tables:
        table_cols = table.columns.to_list()

        # check whether the column names contain a list of tuples
        if isinstance(table_cols[0], tuple):
            # flatten the list of tuples to a list to be able to check the individual elements
            table_cols = [col for double_col in table_cols for col in double_col]

        # remove a specific unicode character from the column names
        cols_clean = [str(col).replace('\u2009', '') for col in table_cols]
        if '9AM' in cols_clean:  # only the detailed table shows data for 9 AM
            detailed = table.copy()
        if any('(7–10 days)' in str(x) for x in cols_clean):  # check if any of the list items contains the substring
            # (only the table including the forecast for the next week contains the string "(7–10 days)")
            week = table.copy()
    return detailed, week


def filter_table(index_col, df_table):
    """ set the value descriptions as index of the DataFrame and filter for the value descriptions of interest

    :param index_col: the name of the column of the DataFrame that is to be set as index (string)
    :param df_table: the DataFrame that is to be filtered (pandas DataFrame)
    :return: the filtered dataset with the index_col as index (pandas DataFrame)
    """
    try:
        # define a column of the DataFrame as index
        df_table.set_index(index_col, inplace=True)
    except KeyError as ke:
        logging.error(f'{str(ke)}, the column name that is to be set as index has most likely changed')
        return None
    else:
        index = df_table.index.to_list()

        # identify the value descriptions that are to be scraped from the website
        items_to_keep = [x for x in index if x in ['Wind km/h', 'Rain mm', 'Temp°C', 'Feels °C', 'Humid Humid %',
                                                   'High', 'Low', 'Feels°C']]
        if not items_to_keep:
            logging.error('No cols of interest for the detailed table were found, cols have most likely been renamed')
            return None
        else:
            # filter for the value descriptions that are to be scraped from the website
            return df_table.loc[items_to_keep].copy()


def get_day_of_interest(day='today'):
    """ generate a string representing the date of a specific day in the format used in the columns of the website's
    tables

    :param day: the day for which the string is to be generated (string, one of 'today', 'tomorrow', 'next_week')
    :return: a string in the format "weekday-date" (e.g., Friday31; string)
    """
    # retrieve the date of the day of interest
    day_date = dt.date.today()
    if day != 'today':
        days_dict = {'tomorrow': 1, 'next_week': 7}
        day_date = day_date + dt.timedelta(days=days_dict[day])

    # get the corresponding weekday of the date of interest
    weekday = day_date.strftime('%A')
    # add a leading zero to the day of the date of interest
    day_with_zero = day_date.strftime('%d')
    return f'{weekday}{day_with_zero}'


def cols_to_numeric(df):
    """ for specific columns of the DataFrame, replace strings with numeric values

    :param df: the DataFrame in which strings are to be replaced with numeric values (pandas DataFrame)
    :return: the DataFrame where strings have been replaced with numeric values for certain columns (pandas DataFrame)
    """
    # delete strings from the wind speed and humidity columns to retain only numerical values
    for col in ['Wind km/h', 'Humid Humid %']:
        df[col] = df[col].map(lambda x: int(re.sub("[^0-9]", "", x)))
    # replace "-" with 0 in the precipitation column
    df['Rain mm'] = df['Rain mm'].replace('-', 0)
    return df


def df_to_arr_of_dtype(df_to_convert, d_type, today=False):
    """ convert the DataFrame to a numpy array with the same data type and the same shape as the datasets stored
     in the HDF5 file

    :param df_to_convert: the DataFrame to convert to a numpy array (pandas DataFrame)
    :param d_type: the target datatype (numpy dtype)
    :param today: if set to true, the DataFrame represents today's weather forecast (bool)
    :return: a numpy array with the target datatype and target shape (numpy array)
    """
    # convert dataframe to a numpy array
    array_day = df_to_convert.to_records(index=False)
    try:
        # change the datatype of the numpy array
        array_day = array_day.astype(d_type)
        shape = array_day.shape

        # check whether there are less than 24 entries in the case of today's forecast
        if today and shape[0] != 24:
            number_missing = 24 - shape[0]  # calculate how many entries are missing
            fill_data = np.repeat(-111, number_missing).astype(d_type)  # generate dummy data for the missing entries
            array_day = np.append(fill_data, array_day)  # insert the dummy data into the beginning of the array
            shape = array_day.shape  # get the new shape of the array

        # reshape the array, so it can be appended to the datasets in the HDF5 file
        return array_day.reshape(1, shape[0])
    except TypeError as te:
        logging.error(str(te), 'it was not possible to convert the weather forecast to an array to store it')
        return None


def df_detailed_to_arr(df_both_days, day):
    """ retrieve the data for the day of interest from the table and convert the DataFrame to a numpy
    array that can be saved in the HDF5 file

    :param df_both_days: a DataFrame containing the weather forecasts for today and tomorrow (pandas DataFrame)
    :param day: the day of interest (string, one of 'today', 'tomorrow', or 'next_week')
    :return: a numpy array that is ready to be saved in the HDF5 file (numpy array)
    """
    day_info = get_day_of_interest(day)
    try:
        # retain only the day of interest and transpose the DataFrame so that rows represent the different
        # times of day and columns represent the different measures (e.g., wind speed, precipitation)
        df_day = df_both_days[day_info].transpose().copy()
        # replace strings with numeric values within certain columns
        df_day = cols_to_numeric(df_day)
    except KeyError as ke:
        logging.error(f'the day {str(ke)} was not found in the DataFrame')
    else:
        # convert the DataFrame to a numpy array with the target datatype and shape
        dtype_day = np.dtype([('wind_speed', np.int16), ('precipitation', np.float16), ('temp', np.int16),
                              ('felt_temp', np.int16), ('humidity', np.int16)])
        today = True if day == 'today' else False
        return df_to_arr_of_dtype(df_day, dtype_day, today)


def save_forecast(array_day, dset_name, file_name):
    """ save the weather forecast for a specific day in the HDF5 file of interest

    :param array_day: the numpy array containing the weather forecast (numpy array)
    :param dset_name: the name of the HDF5 dataset in which to store the numpy array (string)
    :param file_name: the name of the HDF5 file in which the HDF5 dataset is stored (string)
    """
    path_to_dset = f'/weather/{dset_name}'
    save_data_to_hdf5(array_day, path_to_dset, 'save weather forecast', file_name)


def save_forecasts_this_week(table, file_name):
    """ clean today's and tomorrow's weather forecasts and save them in the HDF5 file with the specified name

    :param table: the table which contains the detailed forecasts for today and tomorrow (pandas DataFrame)
    :param file_name: the name of the HDF5 file in which the forecasts are to be stored (string)
    """
    # filter for the values of interest
    df_filtered = filter_table(('°C', '°F'), table)
    if df_filtered is not None:
        # convert the DataFrame to an array with the expected datatype and shape
        array_today = df_detailed_to_arr(df_both_days=df_filtered, day='today')
        array_tomorrow = df_detailed_to_arr(df_both_days=df_filtered, day='tomorrow')

        # save the values in the HDF5 file
        if array_today is not None:
            save_forecast(array_today, 'today', file_name)
        if array_tomorrow is not None:
            save_forecast(array_tomorrow, 'tomorrow', file_name)


def save_forecast_next_week(table, file_name):
    """ clean next week's weather forecast and save it in the HDF5 file with the specified name

    :param table: the table which contains the forecasts for next week (pandas DataFrame)
    :param file_name: the name of the HDF5 file in which the forecasts are to be stored (string)
    """
    # filter the table for the day in seven days from now
    cols = table.columns.to_list()
    df_filtered = filter_table(cols[0], table)

    if df_filtered is not None:
        # retrieve all the columns that contain the 10-Day Paris Weather forecast and the day of interest
        day_info = get_day_of_interest('next_week')
        str_of_interest = '10 Day Paris Weather'
        cols_of_interest = [n for n in cols if str_of_interest in str(n) and day_info in str(n)]
        if not cols_of_interest:
            logging.error(f'{str_of_interest} and/or {day_info} not found the data_frame')
        else:
            try:
                # filter for the day of interest and the first value in the columns of interest
                df_next_week = df_filtered[cols_of_interest[0][0]][day_info].copy()

                # transpose the DataFrame so that rows represent the different times of day and columns
                # represent the different measures (e.g., wind speed, precipitation)
                df_next_week = df_next_week.transpose()

                # replace strings with numeric values in certain columns of the DataFrame
                df_next_week = cols_to_numeric(df_next_week)
            except KeyError as e:
                logging.error(
                    f'{str(e)}, the column structure of the next week\'s forecast table has most likely '
                    f'changed')
            else:
                # convert the DataFrame to a numpy array with the target datatype and shape
                dt_next_week = np.dtype(
                    [('wind_speed', np.int16), ('precipitation', np.float16), ('highest_temp', np.int16),
                     ('lowest_temp', np.int16), ('felt_temp', np.int16), ('humidity', np.int16)])
                array_next_week = df_to_arr_of_dtype(df_next_week, dt_next_week)

                # save the values in the HDF5 file
                if array_next_week is not None:
                    save_forecast(array_next_week, 'next_week', file_name)


# determine the tables with the relevant information (this is done based on text data because it can
# be assumed that the order of the tables on the page can change) and differentiating html tags were not found
def scrape_weather_forecasts(url, file_name):
    """ scrape weather forecasts for today, tomorrow and in seven days from the specified website and save
     the scraped data in a HDF5 file with the specified name

    :param url: the website's URL (string)
    :param file_name: the file name of the HDF5 file (string)
    """
    # get all tables from the website, as the relevant data is stored in the tables
    table_forecasts = retrieve_content(url)
    if table_forecasts is not None:
        # get detailed data for today and tomorrow as well as a forecast for next week
        detailed_table, week_table = get_detailed_and_week_table(table_forecasts)

        # retrieve today's and tomorrow's forecast
        if detailed_table is not None:
            save_forecasts_this_week(detailed_table, file_name)
        else:  # if detailed_table is None
            logging.error('it was not possible to retrieve the detailed table for daily weather forecasts')

        # retrieve next week's forecast
        if week_table is not None:
            save_forecast_next_week(week_table, file_name)
        else:
            logging.error('it was not possible to retrieve the table for the 7-day weather forecasts')

    else:  # if it was not possible to retrieve any tables from the website
        logging.error('it was not possible to retrieve tables from the weather forecasts website')


if __name__ == '__main__':
    set_logging_Config('scrape_disneyland.log')
    url_forecasts = 'https://www.weather-forecast.com/locations/Paris/forecasts/latest'
    scrape_weather_forecasts(url_forecasts, 'disneyland.hdf5')
