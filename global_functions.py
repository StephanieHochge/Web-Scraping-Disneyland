from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import h5py
import os
import datetime as dt

import logging


def retrieve_content(url, retrieve_tables=True):
    """ parse the content of a website and return the respective content

    :param url: the url of the website (string)
    :param retrieve_tables: if set to True, only html tables are retrieved from the page, if set to False,
     BeautifulSoup is used to retrieve the content from the website (bool)
    :return: the content retrieved from the website (either list of pandas DataFrames or a BeautifulSoup object)
    """
    try:
        if retrieve_tables:
            # retrieve the website's html tables
            content = pd.read_html(url)  # see Navlani et al. (2021, p. 197)
        else:
            # parse the website's content
            html = requests.get(url)
            content = BeautifulSoup(html.content, 'lxml')  # lxml is used for reasons described in Mitchell (2018, p. 9)
    except HTTPError as e:
        logging.error(str(e))
        return None
    except URLError as e:
        logging.error(str(e))
        return None
    else:
        return content


def get_path_to_file(file):
    """get the file path of a file

    :param file: the file for which to return the file path (string)
    :return: the file path to the file (string)
    """
    # the whole path name is important for the cron job to function properly
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, file)


def add_data(dset, data_array):
    """ append new data to a dataset

    :param dset: the dataset to which the new data is to be appended (HDF5 dataset)
    :param data_array: the new data (numpy array)
    """
    # resize the dataset to be able to append to it
    dset.resize((dset.shape[0] + data_array.shape[0]), axis=0)
    # append the data to the dataset
    dset[-data_array.shape[0]:] = data_array


def save_data_to_hdf5(data_to_save, path_to_dset, save_text, file_name,
                      save_time=np.array([dt.datetime.now().isoformat()], dtype='S26')):
    """ append the new data to the datasets in the HDF5 file and save the storage times

    :param data_to_save: the data that is to be appended (numpy array)
    :param path_to_dset: the path to the dataset to which the data is to be appended (string)
    :param save_text: the message that is to be logged when saving was successful (string)
    :param file_name: the name of the HDF5 file (string)
    :param save_time: the time the data is saved to the datasets in the HDF5 file (numpy array)
    """
    # get the path to the HDF5 File and the dataset storing the storage times
    file_hdf = get_path_to_file(file_name)
    path_times = f'{path_to_dset}_times'

    try:
        with h5py.File(file_hdf, 'r+') as f:
            # save the values and the storage times in the HDF5 file
            dset = f[path_to_dset]
            dset_time = f[path_times]
            add_data(dset, data_to_save)
            add_data(dset_time, save_time)
    except FileNotFoundError:
        logging.error('the hdf5 file was not found')
    else:
        logging.info(save_text)


def set_logging_Config(name_logfile):
    """ set the logging configuration (use an existing logfile)

    :param name_logfile: the name of the existing logfile (string)
    """
    path_to_file = get_path_to_file(name_logfile)
    logging.basicConfig(filename=path_to_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %('
                                                                           'message)s')


def create_waits_csv(csv_name):
    """ create a csv file that serves as a short term storage for the wait times

    :param csv_name: the name of the csv file (string)
    """
    # define the data structure of the csv file
    data_structure = {
        'princess_pavilion': [],
        'meet_mickey': [],
        'peter_pans_flight': [],
        'big_thunder_mountain': [],
        'orbitron': [],
        'storage_times': []
    }

    # create a DataFrame with the defined data structure and save the DataFrame as a csv file
    df = pd.DataFrame(data_structure)
    df.to_csv(csv_name, mode='w', index=False, header=True)
