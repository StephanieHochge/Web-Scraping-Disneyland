# load packages
import datetime as dt
import logging

import numpy as np
import pandas as pd

from global_functions import retrieve_content, set_logging_Config, save_data_to_hdf5, get_path_to_file


def save_wait_times(ride, df_wait_times, rides_dict, website, file_name):
    """ save the wait time for a ride in the respective dataset in the HDF5 file with the specified name

    :param ride: the ride for which the wait time is to be saved (string)
    :param df_wait_times: the DataFrame containing the wait times for all five rides (pandas DataFrame)
    :param website: the website from which the wait times were retrieved (string)
    :param rides_dict: a dictionary that translates the original names of the rides into the names under
    which they are stored in the HDF5 file (dictionary of strings)
    :param file_name: the name of the HDF5 file (string)
    """
    # create a numpy array with the wait time for the ride to be able to save this in the HDF5 dataset
    val_to_save = np.array([int(df_wait_times.loc[ride, 'WAIT TIME'])])
    path_to_dset = f'/rides/{rides_dict[ride]}/{website}'
    save_text = f'stored wait times from {website}'
    save_data_to_hdf5(val_to_save, path_to_dset, save_text, file_name)


def get_wait_times(tables):
    """look for the table containing wait times (if e.g., the website owner would decide to add more tables)

    :param tables: a list of DataFrames found on the website (list of pandas DataFrames)
    :return: the table displaying wait times (pandas DataFrame)
    """
    for table in tables:
        cols = table.columns.to_list()
        if any('wait time' in col.lower() for col in cols):
            return table
        else:
            logging.error('the table of interest was not found, because no column contained the term "wait times"')
            return None


def wait_col_to_numeric(df_wait_times):
    """ make sure that all cells in the DataFrame containing wait times are numerical

    :param df_wait_times: the DataFrame that contains wait times (pandas DataFrame)
    :return: a DataFrame with numerical values only (pandas DataFrame)
    """
    # extract numerical values from cells
    df_wait_times['WAIT TIME'] = df_wait_times['WAIT TIME'].str.extract('(\d+)')
    # fill missing values with -1
    df_wait_times['WAIT TIME'].fillna(-1, inplace=True)
    return df_wait_times


def restructure_df_waits(df_waits, rides_dict):
    """ restructure the DataFrame that stores the wait times so that it can be appended to the csv file
    that serves as the short term storage for the wait times

    :param df_waits: the DataFrame storing the wait times that is to be restructured (pandas DataFrame)
    :param rides_dict: a dictionary that translates the original names of the rides into the names of the
    columns in the csv file (dictionary of strings)
    :return: the restructured DataFrame that can be appended to the csv file (pandas DataFrame)
    """
    # transpose the DataFrame so that the wait times for the rides are stored in the columns
    df_transposed = pd.DataFrame(df_waits['WAIT TIME']).transpose().copy()
    # filter for the rides of interest
    df_filtered = df_transposed[rides_dict.keys()].copy()
    # rename the columns so that they more closely resemble the names of the datasets in the HDF5 file
    df_filtered.rename(columns=rides_dict, inplace=True)
    # add a column saving the time when the data was scraped from the website as storage time
    df_filtered['storage_time'] = [dt.datetime.now().isoformat()]
    return df_filtered


def save_waits_to_csv(df_waits, rides_dict, csv_file):
    """ append the wait times stored in the DataFrame to the specified csv file

    :param df_waits: the DataFrame storing the wait times (pandas DataFrame)
    :param rides_dict: a dictionary that translates the original names of the rides into the names of the
    columns in the csv file (dictionary of strings)
    :param csv_file: the name of the csv file in which the wait times are to be stored (string)
    """
    # restructure the DataFrame so that it conforms to the data structure of the csv file
    csv_path = get_path_to_file(csv_file)
    df_to_save = restructure_df_waits(df_waits, rides_dict)
    df_to_save.to_csv(csv_path, mode='a', index=False, header=False)


def scrape_q_park(csv_file, rides_dict):
    """ scrape the wait times for the specified rides from the queue park website and store them in the csv file

    :param csv_file: the name of the csv file (string)
    :param rides_dict: a dictionary that translates the original names of the rides into the names of the
    columns in the csv file (dictionary of strings)
    """
    url_queue_park = 'https://queue-park.com/parks/disneyland/disneylandparis-park/'
    queue_tables = retrieve_content(url_queue_park)

    if queue_tables is not None:
        df_waits_all = get_wait_times(queue_tables)
        if df_waits_all is not None:
            # clean the table (currently, I don't differentiate between 'broken down' and 'closed')
            try:
                # define the ride names as index of the dataset
                df_waits_all.set_index('ATTRACTION', inplace=True)
                # extract the numerical values from the wait time column
                df_waits_all = wait_col_to_numeric(df_waits_all)
            except KeyError as e:
                logging.error(f'the keys "ATTRACTION" and/or "WAIT TIME" do not exist in the data frame: {str(e)}')
            else:
                # save the wait times in the csv file
                save_waits_to_csv(df_waits_all, rides_dict, csv_file)
    else:  # if queue_tables is None
        logging.error('it was not possible to retrieve tables from the queue-park website')


def scrape_q_times(csv_file, rides_dict):
    """ scrape the wait times for the specified rides from the queue times website and store them in the csv file

    :param csv_file: the name of the csv file (string)
    :param rides_dict: a dictionary that translates the original names of the rides into the names under
    which they are stored in the HDF5 file (dictionary of strings)
    """
    # scrape data from the queue times website (see Broucke et al. (2018, p. 199))
    url_queue_times = 'https://queue-times.com/parks/4/queue_times'
    # parse the website's content
    soup_times = retrieve_content(url_queue_times, retrieve_tables=False)

    # die einzelnen Ride Namen haben immer die Klasse class='has-text-weight-normal'
    # die zugehörigen Wartezeiten haben immer die Klasse class='has-text-weight-bold'
    if soup_times is not None:
        # wait times are stored in a CSS class called 'panel-block'
        soup_panels = soup_times.find_all('a', class_='panel-block')
        if soup_panels:
            ride_names = []  # initialize a list for the ride names
            wait_times = []  # initialize a list for the wait times
            for ride in soup_panels:
                # the individual ride names have the class 'has-text-weight-normal'
                # the associated wait times have the class 'has-text-weight-bold'
                ride_name_html = ride.find('span', class_='has-text-weight-normal')
                wait_time_html = ride.find('span', class_='has-text-weight-bold')
                if ride_name_html and wait_time_html:  # both information must exist in the panel-block, because
                    # otherwise it is not a ride and instead e.g., the name of the topic land
                    ride_name = ride_name_html.get_text(strip=True)
                    ride_time = wait_time_html.get_text(strip=True)
                    ride_names.append(ride_name)
                    wait_times.append(ride_time)
                else:
                    continue

            # check whether the number of retrieved ride_names is the same as the number of retrieved wait times
            if len(ride_names) == len(wait_times):
                # create a DataFrame with the ride names as index and the wait times as column
                df_waits = pd.DataFrame(wait_times, index=ride_names, columns=['WAIT TIME'])
                # extract the numerical values from the wait time column
                df_waits = wait_col_to_numeric(df_waits)
                # save the wait times in the csv file
                save_waits_to_csv(df_waits, rides_dict, csv_file)

            else:  # if the number of extracted ride names does not equal the number of extracted wait times
                logging.error('the number of ride_names does not equal the number of wait times, '
                              'presumably the class names have changed')
        else:  # if not soup_panels
            logging.error('the soup_panels variable is empty, presumably the class panel-block does not exist anymore')
    else:  # if soup times is None
        logging.error('it was not possible to retrieve the content from the queue-times website')


if __name__ == '__main__':
    set_logging_Config('scrape_disneyland.log')

    # create a dictionary for the rides that translates the original names of the rides into the names under
    # which they are stored in the HDF5 and csv file
    rides = {'Princess Pavilion': 'princess_pavilion', 'Meet Mickey Mouse': 'meet_mickey',
             'Peter Pan\'s Flight': 'peter_pans_flight', 'Big Thunder Mountain': 'big_thunder_mountain',
             'Orbitron®': 'orbitron'}

    # save the wait times in a csv file (data is first aggregated in a csv file, which is transferred to the HDF5
    # file once per day)
    scrape_q_park('waits_q_park.csv', rides)
    scrape_q_times('waits_q_times.csv', rides)

    print('I scraped data')
