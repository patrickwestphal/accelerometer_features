from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from accelerometerfeatures.utils import pairwise_iterator


class Interpolator(object):
    """
    Performs interpolation on data frames which hold time-stamped data points.
    Hence, it is assumed that a column called `timestamp` exists.
    """
    def __init__(
            self,
            data_frame: pd.DataFrame,
            target_sample_frequency_in_hz = 16,
            biggest_acceptable_gap_size_in_no_samples = 10):

        assert 'timestamp' in data_frame.columns
        self.data_frame = data_frame
        self.target_sample_frequency_in_hz = target_sample_frequency_in_hz

        self.biggest_acceptable_gap_size_in_no_samples = \
            biggest_acceptable_gap_size_in_no_samples

        self.sample_time_delta_in_secs = \
            1.0 / self.target_sample_frequency_in_hz

        # convert datetime data into float timestamps, e.g. 1528266608.065
        self.timestamps = \
            self.data_frame.timestamp.transform(datetime.timestamp)
        self.ignored_data_columns = []

    def get_acceptable_data_shreds_timestamps(self):
        """
        For the data frame `self.data_frame` this methods looks for value gaps
        that are too big to be acceptable. If there is such a gap the data
        frame shall be cut right there and should later be handled as if there
        were two data sets. To do so this method calculates the lists of
        timestamps that belong to each such data set shred.
        """
        biggest_acceptable_gap_in_secs = \
            self.sample_time_delta_in_secs * \
            self.biggest_acceptable_gap_size_in_no_samples

        # If there is a gap bigger than the stated biggest acceptable gap size,
        # interpolation doesn't make sense anymore. Thus, the overall data set
        # is cut on those gaps and each part will be treated separately for
        # windowing.
        # The cut indexes are at those points *after* the gap!
        cut_indexes = list(
            self.timestamps.index[
                self.timestamps.diff() > biggest_acceptable_gap_in_secs])

        cut_indexes = [0] + cut_indexes
        cut_indexes.append(len(self.timestamps))

        # [0      1.528266608065000e+09
        #  1      1.528266608133000e+09
        #  ...
        #  Name: timestamp, dtype: float64, 1117    1.52826671978e+09
        #  1118    1.528266719782000e+09
        #  1119    1.528266719831000e+09
        #  ...
        #  Name: timestamp, Length: 3304, dtype: float64, 4561 1.52826694310e+09
        #  4562     1.528266943115000e+09
        #  4563     1.528266943223000e+09
        #  ...
        #  Name: timestamp, Length: 14556, dtype: float64]
        sub_dataset_timestamps_list = \
            [self.timestamps[start:end]
                for start, end in pairwise_iterator(cut_indexes)]

        return sub_dataset_timestamps_list

    def _dbg_get_shred_data(self, data_shred_timestamps):
        start_idx = data_shred_timestamps.first_valid_index()

        start_timestamp = \
            datetime.fromtimestamp(data_shred_timestamps[start_idx])

        end_timestamp = datetime.fromtimestamp(
            data_shred_timestamps[start_idx + len(data_shred_timestamps) - 1])

        return self.data_frame[np.logical_and(
            self.data_frame.timestamp >= start_timestamp,
            self.data_frame.timestamp <= end_timestamp)]

    def get_interpolated_data(self):
        # convert datetime data into float timestamps, e.g. 1528266608.065
        timestamps = self.data_frame.timestamp.transform(datetime.timestamp)

        data_shreds_timestamps = self.get_acceptable_data_shreds_timestamps()

        result_data_frames = []
        column_names = \
            [c for c in self.data_frame.columns
             if c != 'timestamp' and c not in self.ignored_data_columns]

        data_shred_idx = 0

        for data_shred_timestamps in data_shreds_timestamps:
            # Example value for data_shreds_timestamps:
            #
            # 0     1.523357e+09
            # 1     1.523357e+09
            # 2     1.523357e+09
            # 3     1.523357e+09
            #           ...
            # 71    1.523357e+09
            # 72    1.523357e+09
            # 73    1.523357e+09
            # 74    1.523357e+09
            # Name: timestamp, Length: 75, dtype: float64

            if data_shred_timestamps.empty or len(data_shred_timestamps) < 2:
                # Ignored since not meaningful for later processing
                continue

            data_shred_idx += 1

            start_idx = data_shred_timestamps.index[0]
            end_idx = data_shred_timestamps.index[-1]

            # np.arange( )  does not include the stop element! I.e.
            # np.arange(1, 5, 1) --> array([1, 2, 3, 4]) , so without 5
            target_sample_timestamps = np.arange(
                timestamps[start_idx],
                timestamps[end_idx],
                self.sample_time_delta_in_secs,
                np.float)

            target_sample_datetime_timestamps = \
                [dt for dt in
                 map(datetime.fromtimestamp, target_sample_timestamps)]

            data_frame_data = {
                'timestamp': target_sample_datetime_timestamps
            }

            for column_name in column_names:
                whole_column_data = self.data_frame[column_name]
                series = whole_column_data[np.logical_and(
                    whole_column_data.index >= start_idx,
                    whole_column_data.index <= end_idx)]

                interpolate = interp1d(data_shred_timestamps, series)

                # interpolated_series is a numpy array
                interpolated_series = interpolate(target_sample_timestamps)
                data_frame_data[column_name] = interpolated_series

            result_data_frames.append(pd.DataFrame.from_dict(data_frame_data))

        return result_data_frames
