from datetime import datetime
from datetime import timedelta
from random import random, Random
from unittest import TestCase

import pandas as pd

from utils.interpolation import Interpolator

G = 9.81
SEED = 23
GAUSS = Random(SEED).gauss


class TestInterpolation(TestCase):
    @staticmethod
    def _rnd_delta(frequency):
        """
        Returns a random difference (delta) between two time points
        """
        sigma = 0.03
        mu = 1.0 / frequency

        return max(0.001, GAUSS(mu, sigma))

    @staticmethod
    def _rnd_x_y_z():
        x = 3 * G * random()
        y = 3 * G * random()
        z = 3 * G * random()

        return x, y, z

    def _gen_data(
            self, num_entries, approx_frequency_in_hz, num_gaps=0, gap_size=10):

        timestamps = []
        xs = []
        ys = []
        zs = []
        timestamp = datetime.now()

        for i in range(num_entries):
            delta = timedelta(self._rnd_delta(approx_frequency_in_hz) / 100000)
            timestamp = timestamp + delta
            timestamps.append(timestamp)
            x, y, z = self._rnd_x_y_z()
            xs.append(x)
            ys.append(y)
            zs.append(z)

        assert num_gaps * gap_size < num_entries

        res_timestamps = timestamps[:]
        res_xs = xs[:]
        res_ys = ys[:]
        res_zs = zs[:]

        num_shreds = num_gaps + 1

        for gap_no in range(num_gaps, 0, -1):
            gap_median_idx = round(gap_no * num_entries / num_shreds)
            gap_start_idx = gap_median_idx - round(gap_size / 2)

            for idx in range(gap_start_idx + gap_size, gap_start_idx + 1, -1):
                res_timestamps.pop(idx)
                res_xs.pop(idx)
                res_ys.pop(idx)
                res_zs.pop(idx)

        return pd.DataFrame.from_dict({
            'timestamp': res_timestamps,
            'x': res_xs,
            'y': res_ys,
            'z': res_zs
        })

    def test_gaps_are_detected_correctly_01(self):
        """
        Should return the whole data set since there are no gaps
        """
        biggest_acceptable_gap_size_in_no_samples = 10
        target_frequency_in_hz = 16
        actual_approx_frequency_in_hz = 16
        num_data_samples = 1000

        df = self._gen_data(num_data_samples, actual_approx_frequency_in_hz)

        interpolator = Interpolator(
            df,
            target_frequency_in_hz,
            biggest_acceptable_gap_size_in_no_samples)

        shreds_timestamps = interpolator.get_acceptable_data_shreds_timestamps()

        self.assertEqual(1, len(shreds_timestamps))
        self.assertEqual(num_data_samples, len(shreds_timestamps[0]))

    def test_gaps_are_detected_correctly_02(self):
        """
        Should return four data sets since there are three bigger gaps in the
        data
        """
        biggest_acceptable_gap_size_in_no_samples = 10
        target_frequency_in_hz = 16
        actual_approx_frequency_in_hz = 16
        num_data_samples = 1000
        num_gaps = 3
        expected_num_shreds = num_gaps + 1

        df = self._gen_data(
            num_data_samples, actual_approx_frequency_in_hz, num_gaps,
            2 * biggest_acceptable_gap_size_in_no_samples)

        interpolator = Interpolator(
            df,
            target_frequency_in_hz,
            biggest_acceptable_gap_size_in_no_samples)

        shreds_timestamps = interpolator.get_acceptable_data_shreds_timestamps()

        self.assertEqual(expected_num_shreds, len(shreds_timestamps))

    def test_interpolation_is_done_correctly_01(self):
        biggest_acceptable_gap_size_in_no_samples = 10
        target_frequency_in_hz = 16
        actual_approx_frequency_in_hz = 8
        num_data_samples = 1000

        df = self._gen_data(num_data_samples, actual_approx_frequency_in_hz)

        interpolator = Interpolator(
            df, target_frequency_in_hz,
            biggest_acceptable_gap_size_in_no_samples)

        interpolated = interpolator.get_interpolated_data()

        self.assertEqual(1, len(interpolated))

        start_idx = interpolated[0].timestamp.first_valid_index()
        start_datetime = interpolated[0].timestamp[start_idx]
        end_idx = interpolated[0].timestamp.last_valid_index()
        end_datetime = interpolated[0].timestamp[end_idx]
        delta_in_seconds = (end_datetime - start_datetime).total_seconds()
        expected_num_entries = (delta_in_seconds * target_frequency_in_hz) + 1

        # The calculated number of entries should be a round number without a
        # fraction
        self.assertTrue(expected_num_entries.is_integer())

        self.assertEqual(expected_num_entries, len(interpolated[0]))

    def test_interpolation_is_done_correctly_02(self):
        biggest_acceptable_gap_size_in_no_samples = 10
        target_frequency_in_hz = 16
        actual_approx_frequency_in_hz = 32
        num_data_samples = 1000

        df = self._gen_data(num_data_samples, actual_approx_frequency_in_hz)

        interpolator = Interpolator(
            df, target_frequency_in_hz,
            biggest_acceptable_gap_size_in_no_samples)

        interpolated = interpolator.get_interpolated_data()

        self.assertEqual(1, len(interpolated))

        start_idx = interpolated[0].timestamp.first_valid_index()
        start_datetime = interpolated[0].timestamp[start_idx]
        end_idx = interpolated[0].timestamp.last_valid_index()
        end_datetime = interpolated[0].timestamp[end_idx]
        delta_in_seconds = (end_datetime - start_datetime).total_seconds()
        expected_num_entries = (delta_in_seconds * target_frequency_in_hz) + 1

        # The calculated number of entries should be a round number without a
        # fraction
        self.assertTrue(expected_num_entries.is_integer())

        self.assertEqual(expected_num_entries, len(interpolated[0]))
