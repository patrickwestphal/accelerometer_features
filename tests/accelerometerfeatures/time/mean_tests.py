from datetime import datetime
from unittest.case import TestCase

import pandas as pd

from accelerometerfeatures.time import mean


class TestMean(TestCase):
    def test_x_y_z_mean(self):
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

        mean_data = (-1.152, 2.266666666666667, -1.440666666666667)

        self.assertAlmostEqual(mean.from_df(acc_data), mean_data, places=8)

    def test_magnitude_mean(self):
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

        mean_data = 10.207554961123941,

        self.assertAlmostEqual(
            mean.from_df(magnitude_data), mean_data, places=8)
