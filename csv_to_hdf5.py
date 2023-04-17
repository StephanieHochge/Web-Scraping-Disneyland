import numpy as np
import pandas as pd

from global_functions import create_waits_csv, get_path_to_file, save_data_to_hdf5, set_logging_Config


def read_waits_from_csv(csv_name):
    """ read the wait times stored in the csv file with the specified name and transfer them into a DataFrame

    :param csv_name: the name of the csv file (string)
    :return: a DataFrame with the wait times read from the csv file (pandas DataFrame)
    """
    df_park = pd.read_csv(csv_name, index_col='storage_times')
    # delete the name of the index
    df_park.index.name = None
    return df_park


def transfer_df_to_hdf5(df_waits, website, hdf5_file):
    """ transfer the data from the specified DataFrame to the dataset in the HDF5 file that stores the data from
    the specified website

    :param df_waits: the DataFrame holding the data to be stored (pandas DataFrame)
    :param website: the name of the website from which the data was taken (string)
    :param hdf5_file: the name of the HDF5 file in which to store the data (string)
    """
    # transfer the data of each column to the HDF5 file
    for col in df_waits.columns.to_list():
        data_to_save = np.array([int(x) for x in df_waits[col].to_list()])
        path_to_dset = f'/rides/{col}/{website}'
        save_text = f'stored wait times from {website}'  # the text to log
        save_time = np.array(df_waits.index.to_list(), dtype='S26')
        save_data_to_hdf5(data_to_save, path_to_dset, save_text, hdf5_file, save_time)


def transfer_csv_to_hdf5(website, csv_name, hdf5_file):
    """ read data from the csv file with the specified name and write the data to the dataset in the HDF5
    file that stores the data from the specified website

    :param website: the name of the website from which the data was taken (string)
    :param csv_name: the name of the csv file (string)
    :param hdf5_file: the name of the HDF5 file in which to store the data (string)
    """
    df_waits = read_waits_from_csv(csv_name)
    transfer_df_to_hdf5(df_waits, website, hdf5_file)


if __name__ == '__main__':
    set_logging_Config('scrape_disneyland.log')
    websites = ['q_park', 'q_times']
    for site in websites:
        # read the data from the csv files and transfer them to the respective datasets in the HDF5 file
        csv_file = get_path_to_file(f'waits_{site}.csv')
        transfer_csv_to_hdf5(site, csv_file, 'disneyland.hdf5')
        # recreate the csv files and delete the previously stored data
        create_waits_csv(csv_file)
