import h5py
import pandas as pd
import pytest
from hdf5_data_model import create_hdf5_file
from scrape_weather import scrape_weather_forecasts
from global_functions import create_waits_csv
from scrape_rides import scrape_q_park, scrape_q_times
from csv_to_hdf5 import transfer_csv_to_hdf5
import time


class TestDataModel:
    """ a test class to test aspects of the defined data model for scraping wait times and weather forecasts """

    def test_number_objects(self):
        """ test whether the expected number of objects (datasets, groups) was generated """
        create_hdf5_file('test_scraping', 'w')
        with h5py.File('test_scraping.hdf5', 'r') as f:
            assert len(f.keys()) == 2
            weather = f['weather']
            assert len(weather.keys()) == 6
            rides = f['rides']
            assert len(rides.keys()) == 5

    def test_data(self):
        """ test whether the generated datasets have the expected shapes and metadata """
        create_hdf5_file('test_scraping', 'w')
        with h5py.File('test_scraping.hdf5', 'r') as f:
            dset_park = f['rides/princess_pavilion/q_park']
            dset_times = f['rides/peter_pans_flight/q_times']
            dset_today = f['weather/today']
            dset_next_week = f['weather/next_week']
            assert dset_park.attrs['topic_area'] == 'Fantasyland'
            assert dset_times.attrs['name_ride'] == 'Peter Pan\'s Flight'
            assert dset_today.attrs['time_delta'] == 1
            assert dset_next_week.attrs['time_delta'] == 8
            assert dset_park.shape == (2,)
            assert dset_times.shape == (2,)
            assert dset_today.shape == (2, 24)
            assert dset_next_week.shape == (2, 3)

    def test_dimension_scales(self):
        """ test whether the addition of the storage time as dimension scales to the datasets works """
        create_hdf5_file('test_scraping', 'w')
        with h5py.File('test_scraping.hdf5', 'r') as f:
            dset_park = f['rides/princess_pavilion/q_park']
            dset_next_week = f['weather/next_week']
            assert list(dset_park.dims[0]['storage_time'][:2]) == [b'1800-01-01T00:00:00.000000',
                                                                   b'1800-01-02T00:00:00.000000']
            assert list(dset_next_week.dims[0]['storage_time'][:2]) == [b'1800-01-01T00:00:00.000000',
                                                                        b'1800-01-02T00:00:00.000000']


class TestScraping:
    """ a test class to test whether scraping the three websites (the weather forecasts and the wait times) works """

    def test_weather_scraping(self):
        """ test whether scraping the weather forecasts works """
        # create a new HDF5 file for testing purposes
        create_hdf5_file('test_scraping', 'w')
        url_forecasts = 'https://www.weather-forecast.com/locations/Paris/forecasts/latest'

        # scrape the three weather forecasts (forecasts for today, tomorrow and next week)
        scrape_weather_forecasts(url_forecasts, 'test_scraping.hdf5')

        # check whether the datasets in the HDF5 file have the expected shapes
        with h5py.File('test_scraping.hdf5', 'r') as f:
            dset_today = f['weather/today']
            assert dset_today.shape == (3, 24)
            dset_tomorrow = f['weather/tomorrow']
            assert dset_tomorrow.shape == (3, 24)
            dset_next_week = f['weather/next_week']
            assert dset_next_week.shape == (3, 3)

    def test_wait_time_scraping(self):
        """ test whether scraping the wait times works """
        # create a new HDF5 file for testing purposes
        create_hdf5_file('test_scraping', 'w')
        rides = {'Princess Pavilion': 'princess_pavilion', 'Meet Mickey Mouse': 'meet_mickey',
                 'Peter Pan\'s Flight': 'peter_pans_flight', 'Big Thunder Mountain': 'big_thunder_mountain',
                 'Orbitron®': 'orbitron'}
        websites = ['q_times', 'q_park']
        for site in websites:
            # create csv files for testing purposes which serve as a short term storage for the wait times
            create_waits_csv(f'test_waits_{site}.csv')

        # perform web scraping from both websites three times
        for i in range(3):
            scrape_q_times('test_waits_q_times.csv', rides)
            scrape_q_park('test_waits_q_park.csv', rides)
            # wait for three seconds before carrying out the next scrape iteration to not overload the websites
            time.sleep(3)

        for site in websites:
            # check whether the short-term storage in the csv files works
            csv_name = f'test_waits_{site}.csv'
            df_waits = pd.read_csv(csv_name)
            assert df_waits.shape == (3, 6)

            # test transfer from the csv to the HDF5 file by checking whether the HDF5 datasets have the expected shapes
            transfer_csv_to_hdf5(site, csv_name, 'test_scraping.hdf5')
            with h5py.File('test_scraping.hdf5', 'r') as f:
                dset = f[f'rides/big_thunder_mountain/{site}']
                dim = dset.dims[0]['storage_time']
                assert dset.shape == (5,)
                assert dim.shape == (5,)

            # recreate the csv file to delete the data
            create_waits_csv(csv_name)


