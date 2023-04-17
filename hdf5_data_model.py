import h5py
import numpy as np
from global_functions import create_waits_csv


def attach_time_scale(file, path, dummy_data, dtype):
    """ create a dataset to store the storage times of the data and then add it as a dimension scale to the actual
    datasets

    :param file: the HDF5 file in which the dataset is to be stored (HDF5 file)
    :param path: the path to the actual datasets (string)
    :param dummy_data: storage times for initializing the data set, so that further values can be added to it
    later on (numpy array with d_type 'S26')
    :param dtype: the data type of the dummy data (numpy dtype)
    """
    dset_time = file.create_dataset(f'{path}_times', data=dummy_data, maxshape=(None,), dtype=dtype,
                                    chunks=True)
    dset_time.make_scale('storage_time')
    dset_content = file[path]
    dset_content.dims[0].attach_scale(dset_time)


def attach_metadata_rides(ride_name, topic_area, dset, source):
    """ generate metadata for a ride and attach the metadata to the corresponding dataset

    :param ride_name: the name of the ride (string)
    :param topic_area: the topic are the ride belongs to (string)
    :param dset: the HDF5 dataset of the corresponding ride (HDF5 dataset)
    :param source: the website where the data is taken from (string)
    """
    if source == 'q_times':
        website = 'www.queue-times.com'
    else:
        website = 'www.queue-park.com'
    attrs_dict = {'name_ride': ride_name, 'name_park': 'Disneyland Paris', 'topic_area': topic_area,
                  'units_waiting_time': 'min', 'time_delta': '1', 'time_units': 'min', 'source': website,
                  'missing_values': -1}
    for attr in attrs_dict:
        dset.attrs[attr] = attrs_dict[attr]

    # add an attribute indicating the number of data points that are dummy data
    dset.attrs['dummy_data_to_delete'] = dset.shape


def attach_metadata_weather(time_frame, file, path):
    """ generate metadata for a weather forecast and attach the metadata to the corresponding dataset

    :param path: the path to the weather forecast dataset (string)
    :param file: the HDF5 file in which the weather forecast dataset is stored (HDF5 file)
    :param time_frame: the time frame of the forecast (today, tomorrow or next week; string)
    """
    dset = file[path]
    if time_frame == 'next_week':
        time_delta = 8  # the values correspond to time windows of 8 hours
    else:
        time_delta = 1  # the values correspond to time windows of 1 hour
    attrs_dict_weather = {'location': 'paris', 'time_delta': time_delta, 'time_units': 'hours',
                          'source': 'www.weather-forecast.com', 'unit_temp': '°C',
                          'unit_prec': 'mm', 'unit_wind_speed': 'km/h', 'unit_humidity': 'percent',
                          'unit_felt_temp': '°C', 'description': f'weather forecast collected for {time_frame}',
                          'dummy_data_to_delete': dset.shape, 'missing_values': -111}
    for name in attrs_dict_weather:
        dset.attrs[name] = attrs_dict_weather[name]


def create_hdf5_file(name, mode):
    """ create the HDF5 file with the specified name, create groups for the rides and weather forecasts and
    add corresponding records

    :param name: the name of the HDF5 file (string)
    :param mode: the mode in which the file should be opened (string)
    """
    # create file
    f = h5py.File(f'{name}.hdf5', mode)

    # data model for the rides

    # indicate which rides were selected
    # five rides that had the highest waitings times on average according to queue times in 2022 were selected
    rides = {'princess_pavilion': ['Princess Pavilion', 'Fantasyland'],
             'meet_mickey': ['Meet Mickey Mouse', 'Fantasyland'],
             'peter_pans_flight': ['Peter Pan\'s Flight', 'Fantasyland'],
             'big_thunder_mountain': ['Big Thunder Mountain', 'Frontierland'],
             'orbitron': ['Orbitron®', 'Fantasyland']}
    websites = ['q_times', 'q_park']  # the two website sources

    # create dummy data to initialize the datasets and to be able to append to it
    dummy_waits = np.repeat(-1, 2)
    dt_storage_time = np.dtype('S26')  # since the ISO-format of timestamps is of length 26
    dummy_timestamps = np.array(['1800-01-01T00:00:00.000000', '1800-01-02T00:00:00.000000'], dtype=dt_storage_time)

    # create for each ride two datasets (one per website)
    for ride in rides.keys():
        for site in websites:
            path_dset = f'/rides/{ride}/{site}'

            # chunked storage is necessary to be able to resize the dataset
            dset_ride = f.create_dataset(path_dset, data=dummy_waits, maxshape=(None,), dtype='int16', chunks=True)

            # add metadata to the datasets
            attach_metadata_rides(rides[ride][0], rides[ride][1], dset_ride, site)

            # attach a dataset that holds storage times as dimension scale to the ride datasets
            attach_time_scale(f, path_dset, dummy_timestamps, dt_storage_time)

    # data model for temperature forecasts

    # define a compound data type for the weather data (the columns are in the same order as in the table on the website
    # to facilitate data retrieval and storage)
    dt_day = np.dtype([('wind_speed', np.int16), ('precipitation', np.float16), ('temp', np.int16),
                       ('felt_temp', np.int16), ('humidity', np.int16)])

    dt_next_week = np.dtype([('wind_speed', np.int16), ('precipitation', np.float16), ('highest_temp', np.int16),
                             ('lowest_temp', np.int16), ('felt_temp', np.int16), ('humidity', np.int16)])

    # create dummy temperature data with the same shape as the final datasets
    dummy_day = np.repeat(-99, 48).reshape(2, 24)  # 24 measurements per day (1 per hour)
    dummy_day = np.array(dummy_day, dtype=dt_day)
    dummy_next_week = np.repeat(-99, 6).reshape(2, 3)  # 3 measurements per day (1 every 8 hours)
    dummy_next_week = np.array(dummy_next_week, dtype=dt_next_week)

    # create datasets for the weather forecasts (for today, tomorrow and next week)
    time_frames = ['today', 'tomorrow']
    for frame in time_frames:
        f.create_dataset(f'/weather/{frame}', data=dummy_day, maxshape=(None, 24), dtype=dt_day, chunks=True)
    f.create_dataset(f'weather/next_week', data=dummy_next_week, maxshape=(None, 3), dtype=dt_next_week, chunks=True)

    # add the storage time as a dimension scale to the weather datasets and attach metadata
    time_frames.append('next_week')
    for frame in time_frames:
        path_dset = f'weather/{frame}'
        attach_time_scale(f, path_dset, dummy_timestamps, dt_storage_time)
        attach_metadata_weather(frame, f, path_dset)

    # closing the file writes our work to disk
    f.close()


if __name__ == '__main__':
    # create the hdf5 file with the specified data model
    create_hdf5_file('disneyland', 'w')

    # create the csv file for aggregating wait times before storing them in the HDF5 file
    create_waits_csv('waits_q_times.csv')
    create_waits_csv('waits_q_park.csv')
