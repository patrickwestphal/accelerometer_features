from datetime import datetime
from unittest.case import TestCase

import pandas as pd

from accelerometerfeatures.time import stdev


class TestStDev(TestCase):
    def test_x_y_z_stdev(self):
        dt1 = datetime(2000, 1, 2, 12, 34)
        dt2 = datetime(2000, 1, 2, 12, 35)
        dt3 = datetime(2000, 1, 2, 12, 36)

        acc_data = pd.DataFrame(
            [
                [-1.234, 5.678, -9.012, dt1],
                [3.456, -7.890, 1.234, dt2],
                [-5.678, 9.012, 3.456, dt3]],
            columns=['x', 'y', 'z', 'timestamp'],
        )

        stdev_data = (4.567552079615513, 8.952502294517066, 6.650423846141939)

        self.assertAlmostEqual(
            stdev.from_df(acc_data)[0], stdev_data[0], places=8)
        self.assertAlmostEqual(
            stdev.from_df(acc_data)[1], stdev_data[1], places=8)
        self.assertAlmostEqual(
            stdev.from_df(acc_data)[2], stdev_data[2], places=8)

    def test_magnitude_stdev(self):
        dt1 = datetime(2000, 1, 2, 12, 34)
        dt2 = datetime(2000, 1, 2, 12, 35)
        dt3 = datetime(2000, 1, 2, 12, 36)

        magnitude_data = pd.DataFrame(
            [
                [10.722806722122712, dt1],
                [8.701654555313029, dt2],
                [11.198203605936088, dt3]],
            columns=['magnitude', 'timestamp'],
        )

        stdev_data = 1.325632895431829,

        # self.assertAlmostEqual(
        #     stdev.from_df(magnitude_data), stdev_data, places=8)

        self.assertAlmostEqual(
            stdev.from_df(magnitude_data)[0], stdev_data[0], places=8)
