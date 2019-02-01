import datetime
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from accelerometerfeatures.utils.interpolation import Interpolator


class AccelerometerDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self):
        return len(self.windows)


class AccelerometerDatasetLoader(object):
    """
    This class shall serve the following purposes:
    - Read the actual CSV data
    - Build individual data sets for each user
    - Pre-process the accelerometer data contained in the CSV file
      - Apply windowing
      - Optionally apply interpolation to get a stable sample rate

    TODO: Collect statistics (discarded windows/sensor samples)
    TODO: Cut out gaps in non-interpolating mode
    """
    def __init__(
            self,
            csv_file_path,
            window_size_in_seconds=30,
            window_step_size_in_seconds=15,
            perform_interpolation=False,
            interpolation_frequency=16):

        self.csv_file_path = csv_file_path
        self.acc_data = pd.read_csv(self.csv_file_path, parse_dates=[1])
        self.users = list(self.acc_data.user.unique())
        self.dates = list(
            self.acc_data.timestamp.transform(lambda e: e.date()).unique())
        self.perform_interpolation = perform_interpolation
        self.window_size_in_seconds = window_size_in_seconds
        self.window_step_size_in_seconds = window_step_size_in_seconds
        self.interpolation_frequency = interpolation_frequency
        self.min_no_samples_per_window = 10

    def get_user_data(self, user, date=None):
        assert isinstance(user, str)

        if date is not None:
            assert isinstance(date, datetime.date)

        user_data = self.acc_data[self.acc_data.user == user]

        if date is not None:
            date_idxs = \
                user_data.timestamp.transform(lambda t: t.date()) == date
            user_data = user_data[date_idxs]

        user_data.reset_index(drop=True, inplace=True)

        if self.perform_interpolation:
            interpolator = \
                Interpolator(user_data, self.interpolation_frequency, 10)
            interpolator.ignored_data_columns.append('user')
            interpolator.ignored_data_columns.append('class')
            user_data = interpolator.get_interpolated_data()
        else:
            # Just to return data in the same schema the Interpolator object
            # returns.
            # FIXME: This is not generic and requires insight into the data. Should be replaced with something more generic
            user_data = [user_data[['timestamp', 'x', 'y', 'z']]]

        return user_data

    def get_user_data_windows(self, user, date=None):
        win_size = datetime.timedelta(seconds=self.window_size_in_seconds)
        step_size = datetime.timedelta(seconds=self.window_step_size_in_seconds)
        expected_no_samples_per_window = \
            self.window_size_in_seconds * self.interpolation_frequency

        user_data = self.get_user_data(user, date)
        for data_shred in user_data:
            if data_shred.empty:
                continue

            first_idx = data_shred.first_valid_index()
            last_idx = data_shred.last_valid_index()
            last_datetime = data_shred.timestamp[last_idx]

            start_datetime = data_shred.timestamp[first_idx]
            end_datetime = start_datetime + win_size

            while end_datetime <= last_datetime:
                win_idxs = np.logical_and(
                    data_shred.timestamp >= start_datetime,
                    data_shred.timestamp < end_datetime)

                window_data = data_shred[win_idxs]
                window_data.reset_index(drop=True, inplace=True)

                # get window label
                df_idxs = np.logical_and(
                    self.acc_data.timestamp >= start_datetime,
                    self.acc_data.timestamp < end_datetime)
                labels = self.acc_data[df_idxs]['class'].unique()

                # for the next round
                start_datetime = start_datetime + step_size
                end_datetime = start_datetime + win_size

                if not len(labels) == 1:
                    # window contains data with mixed labels --> ignore
                    continue

                label = labels[0]

                if self.perform_interpolation:
                    if len(window_data) < expected_no_samples_per_window:
                        continue
                    else:
                        assert len(window_data) == \
                               expected_no_samples_per_window
                else:
                    if len(window_data) < self.min_no_samples_per_window:
                        continue

                yield window_data, label

    def get_dataset_for_users(self, users: list, date=None):
        all_windows = []

        for user in users:
            for window in self.get_user_data_windows(user, date):
                window_data = np.array([
                    window[0]['x'].as_matrix(),
                    window[0]['y'].as_matrix(),
                    window[0]['z'].as_matrix(),
                ])
                window_label = window[1]

                all_windows.append((window_data, window_label))

        return AccelerometerDataset(all_windows)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('input_file')
    arg_parser.add_argument('user')
    arg_parser.add_argument('--date', metavar='YYYY-mm-dd')
    arg_parser.add_argument('--interpolate', action='store_true', default=False)
    args = arg_parser.parse_args()

    file_path = args.input_file
    user = args.user

    date = args.date
    if date is not None:
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    perform_interpolation = args.interpolate

    loader = AccelerometerDatasetLoader(
        file_path, perform_interpolation=perform_interpolation)

    for window, label in loader.get_user_data_windows(user, date):
        print(window)
        plt.plot(window.timestamp, window.x)
        plt.plot(window.timestamp, window.y)
        plt.plot(window.timestamp, window.z)
        plt.title(label)
        plt.show()
