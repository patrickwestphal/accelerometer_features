import csv
import os
import uuid
from datetime import datetime
from datetime import timedelta
from random import Random
from random import random
from tempfile import TemporaryDirectory
from unittest import TestCase

from accelerometerfeatures.utils.pytorch.dataset import \
    AccelerometerDatasetLoader

G = 9.81
SEED = 123
GAUSS = Random(SEED).gauss


class TestAccelerometerDatasetLoader(TestCase):
    @staticmethod
    def _rnd_delta(frequency):
        """Returns a random difference (delta) between two time points"""
        sigma = 0.03
        mu = 1.0 / frequency

        return max(0.001, GAUSS(mu, sigma))

    @staticmethod
    def _rnd_x_y_z():
        x = 3 * G * random()
        y = 3 * G * random()
        z = 3 * G * random()

        return x, y, z

    def _fill_file_with_generated_data(
            self, file_path, num_users, num_entries_per_user,
            approx_frequency_in_hz):

        with open(file_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['user', 'timestamp', 'x', 'y', 'z', 'class'])

            for _ in range(num_users):
                user_id = uuid.uuid4()
                # data series for all users start at 6 am (to ease debugging)
                timestamp = datetime(2042, 5, 23, 6, 0, 0)

                x, y, z = self._rnd_x_y_z()
                csv_writer.writerow(
                    [user_id, timestamp, x, y, z, 'dummy class'])

                #            -1 since the first entry was already written above
                for i in range(num_entries_per_user - 1):
                    delta = timedelta(
                        seconds=self._rnd_delta(approx_frequency_in_hz))

                    timestamp = timestamp + delta
                    x, y, z = self._rnd_x_y_z()

                    csv_writer.writerow(
                        [user_id, timestamp, x, y, z, 'dummy class'])

    def test_get_user_data_windows_01(self):
        """
        The approximate frequency of the generated samples equals the target
        interpolation frequency.
        """
        tmp_dir = TemporaryDirectory()
        tmp_file_path = os.path.join(
            tmp_dir.name, 'test_get_user_data_windows_01.csv')

        approx_frequency = 16
        target_frequency = 16
        num_users = 5
        num_samples_per_user = 1440

        # Needed to assure the last window is filled completely. If no safety
        # margin is applied it might be that by chance
        # self._fill_file_with_generated_data(  ) will pick too small random
        # distances which leaves the last window unfilled by some milliseconds.
        num_safety_samples_to_fill_last_window = 10

        window_size_in_seconds = 30
        window_step_size_in_seconds = 10
        perform_interpolation = True

        approx_time_span_per_user_in_secs = \
            num_samples_per_user * (1 / approx_frequency)
        approx_num_windows_per_user = \
            int((approx_time_span_per_user_in_secs - window_size_in_seconds) /
                window_step_size_in_seconds) + 1
        expected_entries_per_window = \
            int(window_size_in_seconds * target_frequency)

        self._fill_file_with_generated_data(
            tmp_file_path,
            num_users,
            num_samples_per_user + num_safety_samples_to_fill_last_window,
            approx_frequency)

        data_loader = AccelerometerDatasetLoader(
            tmp_file_path,
            window_size_in_seconds,
            window_step_size_in_seconds,
            perform_interpolation,
            target_frequency)

        users = data_loader.users

        for user in users:
            # Needed since `data_loader.get_user_data_windows(  )` returns a
            # generator
            windows = [w for w in data_loader.get_user_data_windows(user)]

            self.assertEqual(approx_num_windows_per_user, len(windows))

            for window, labels in windows:
                self.assertEqual(expected_entries_per_window, len(window))

    def test_get_user_data_windows_02(self):
        tmp_dir = TemporaryDirectory()
        tmp_file_path = os.path.join(
            tmp_dir.name, 'test_get_user_data_windows_02.csv')

        approx_frequency = 8
        target_frequency = 16
        num_users = 5
        num_samples_per_user = 720

        # Needed to assure the last window is filled completely. If no safety
        # margin is applied it might be that by chance
        # self._fill_file_with_generated_data(  ) will pick too small random
        # distances which leaves the last window unfilled by some milliseconds.
        num_safety_samples_to_fill_last_window = 10

        window_size_in_seconds = 30
        window_step_size_in_seconds = 10
        perform_interpolation = True

        approx_time_span_per_user_in_secs = \
            num_samples_per_user * (1 / approx_frequency)
        approx_num_windows_per_user = \
            int((approx_time_span_per_user_in_secs - window_size_in_seconds) /
                window_step_size_in_seconds) + 1
        expected_entries_per_window = \
            int(window_size_in_seconds * target_frequency)

        self._fill_file_with_generated_data(
            tmp_file_path,
            num_users,
            num_samples_per_user + num_safety_samples_to_fill_last_window,
            approx_frequency)

        data_loader = AccelerometerDatasetLoader(
            tmp_file_path,
            window_size_in_seconds,
            window_step_size_in_seconds,
            perform_interpolation,
            target_frequency)

        users = data_loader.users

        for user in users:
            # Needed since `data_loader.get_user_data_windows(  )` returns a
            # generator
            windows = [w for w in data_loader.get_user_data_windows(user)]

            self.assertEqual(approx_num_windows_per_user, len(windows))

            for window, labels in windows:
                self.assertEqual(expected_entries_per_window, len(window))

    def test_get_user_data_windows_03(self):
        tmp_dir = TemporaryDirectory()
        tmp_file_path = os.path.join(
            tmp_dir.name, 'test_get_user_data_windows_03.csv')

        approx_frequency = 32
        target_frequency = 16
        num_users = 5
        num_samples_per_user = 2880

        # Needed to assure the last window is filled completely. If no safety
        # margin is applied it might be that by chance
        # self._fill_file_with_generated_data(  ) will pick too small random
        # distances which leaves the last window unfilled by some milliseconds.
        num_safety_samples_to_fill_last_window = 10

        window_size_in_seconds = 30
        window_step_size_in_seconds = 10
        perform_interpolation = True

        approx_time_span_per_user_in_secs = \
            num_samples_per_user * (1 / approx_frequency)
        approx_num_windows_per_user = \
            int((approx_time_span_per_user_in_secs - window_size_in_seconds) /
                window_step_size_in_seconds) + 1
        expected_entries_per_window = \
            int(window_size_in_seconds * target_frequency)

        self._fill_file_with_generated_data(
            tmp_file_path,
            num_users,
            num_samples_per_user + num_safety_samples_to_fill_last_window,
            approx_frequency)

        data_loader = AccelerometerDatasetLoader(
            tmp_file_path,
            window_size_in_seconds,
            window_step_size_in_seconds,
            perform_interpolation,
            target_frequency)

        users = data_loader.users

        for user in users:
            # Needed since `data_loader.get_user_data_windows(  )` returns a
            # generator
            windows = [w for w in data_loader.get_user_data_windows(user)]

            self.assertEqual(approx_num_windows_per_user, len(windows))

            for window, labels in windows:
                self.assertEqual(expected_entries_per_window, len(window))
