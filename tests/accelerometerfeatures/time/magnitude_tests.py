from datetime import datetime
from unittest.case import TestCase

import numpy as np
import pandas as pd

from accelerometerfeatures.time.magnitude import from_df


class TestMagnitude(TestCase):
    def test(self):
        dt1 = datetime(2000, 1, 2, 12, 34)
        dt2 = datetime(2000, 1, 2, 12, 35)
        dt3 = datetime(2000, 1, 2, 12, 36)

        acc_data = pd.DataFrame(
            [
                [-1.234, 5.678, -9.012, dt1],
                [3.456, -7.890, 1.234, dt2],
                [-5.678, 9.012, 3.456, dt3]],
            columns=['x', 'y', 'z', 'timestamp'],
            #dtype=[np.float, np.float, np.float, np.datetime64]
        )

        magnitude_data = pd.DataFrame(
            [
                [10.722806722122712, dt1],
                [8.701654555313029, dt2],
                [11.198203605936088, dt3]],
            columns=['magnitude', 'timestamp'],
            #dtype=[]
        )

        self.assertTrue(from_df(acc_data).equals(magnitude_data))
