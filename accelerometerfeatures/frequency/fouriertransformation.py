"""
See https://docs.scipy.org/doc/numpy/reference/routines.fft.html
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from accelerometerfeatures.utils import Window
from accelerometerfeatures.utils import pairwise_iterator


def from_file(file_path, window_size, frequency):
    """
    :param file_path: String containing the file path to the input data file.
    Expected structure: x,y,z,timestamp

    e.g.:

    x,y,z,timestamp
    0.94999998,-9.42000007,1.20000004,2018-10-10 12:54:20.005
    0.44999998,-9.84000015,0.75999999,2018-10-10 12:54:20.067
    0.67000001,-9.96000003,-0.17000000,2018-10-10 12:54:20.229
    1.71000003,-9.27999973,0.34999999,2018-10-10 12:54:20.235
    1.5,-9.4899997,0.72000002,2018-10-10 12:54:20.28
    0.97000002,-9.68000030,-0.10999999,2018-10-10 12:54:20.357
    2.03999996,-9.68000030,0.27000001,2018-10-10 12:54:20.423

    :return: A tuple containing the means per accelerometer dimension
    """

    accel_data = pd.read_csv(file_path, parse_dates=[3])

    return from_df(accel_data, window_size, frequency)


def from_df(dataframe, window_size, frequency):
    """Off-by-one hell"""

    # convert datatime data into float timestamps, e.g. 1528266608.065
    x = dataframe.timestamp.transform(datetime.timestamp)

    step_in_secs = 1. / frequency

    # chosen arbitrarily
    biggest_acceptable_gap_size = 10  # consecutive data points

    # If there is a gap bigger than the stated biggest acceptable gap size,
    # interpolation doesn't make sense anymore. Thus, the overall dataset is
    # cut on those gaps and each part will be treated separately for windowing.
    # The cut indexes are at those points *after* the gap!
    cut_indexes = \
        list(x.index[x.diff() > (step_in_secs * biggest_acceptable_gap_size)])
    cut_indexes = [0] + cut_indexes
    cut_indexes.append(len(x))

    # [0      1.528266608065000e+09
    #  1      1.528266608133000e+09
    #  ...
    #  Name: timestamp, dtype: float64, 1117    1.52826671978e+09
    #  1118    1.528266719782000e+09
    #  1119    1.528266719831000e+09
    #  ...
    #  Name: timestamp, Length: 3304, dtype: float64, 4561     1.52826694310e+09
    #  4562     1.528266943115000e+09
    #  4563     1.528266943223000e+09
    #  ...
    #  Name: timestamp, Length: 14556, dtype: float64]
    sub_dataset_timestamps_list = \
        [x[start:end] for start, end in pairwise_iterator(cut_indexes)]

    frequency_windows = []

    sub_dataset_idx = 0
    for sub_dataset_timestamps in sub_dataset_timestamps_list:
        sub_dataset_idx += 1

        column_names = [c for c in dataframe.columns if c != 'timestamp']
        start_idx = sub_dataset_timestamps.index[0]
        end_idx = sub_dataset_timestamps.index[-1]
        # np.arange( )  does not include the stop element! I.e.
        # np.arange(1, 5, 1) --> array([1, 2, 3, 4]) , so without 5
        x_by_freq = np.arange(
            x[start_idx], x[end_idx], step_in_secs, np.float)

        for column_name in column_names:
            series = \
                dataframe[column_name][start_idx: end_idx+1]  # +1 to incl end

            interpolation = interp1d(sub_dataset_timestamps, series)

            # interpolated_series is a numpy array
            interpolated_series = interpolation(x_by_freq)

            if len(interpolated_series) < window_size:
                first_idx = sub_dataset_timestamps.index[0]
                last_idx = sub_dataset_timestamps.index[-1]
                logging.warning(
                    'Interpolation of sub dataset %i (from %s to %s with %i '
                    'entries) is too small for window size %i' % (
                        sub_dataset_idx,
                        datetime.fromtimestamp(
                            sub_dataset_timestamps[first_idx]).isoformat(),
                        datetime.fromtimestamp(
                            sub_dataset_timestamps[last_idx]).isoformat(),
                        len(interpolated_series),
                        window_size))
                break

            # plt.plot(sub_dataset_timestamps[:-1],
            #          dataframe[column_name][start_idx: end_idx],
            #          x_by_freq, interpolated_series, 'o')
            # plt.title(
            #     'Sub dataset %i, column %s' % (sub_dataset_idx, column_name))
            # plt.show()

            last_window_idx = len(interpolated_series) - window_size
            for window_pos in range(last_window_idx):
                series_window = \
                    interpolated_series[window_pos: window_pos + window_size]

                window_start = datetime.fromtimestamp(x_by_freq[window_pos])
                window_end = \
                    datetime.fromtimestamp(x_by_freq[window_pos+window_size])

                # Documentation:
                # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
                frequency_window = Window(
                    window_start, window_end, np.fft.fft(series_window))
                frequency_windows.append(frequency_window)

    return frequency_windows
