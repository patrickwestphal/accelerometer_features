import numpy as np
import pandas as pd


def from_file(file_path):
    """
    :param file_path: String containing the file path to the input data file.
    Expected structure: x,y,z,timestamp

    e.g.:

    x,y,z,timestamp
    0.949999988079071,-9.420000076293945,1.2000000476837158,2018-10-10 12:54:20.005
    0.44999998807907104,-9.84000015258789,0.7599999904632568,2018-10-10 12:54:20.067
    0.6700000166893005,-9.960000038146973,-0.17000000178813934,2018-10-10 12:54:20.229
    1.7100000381469727,-9.279999732971191,0.3499999940395355,2018-10-10 12:54:20.235
    1.5,-9.489999771118164,0.7200000286102295,2018-10-10 12:54:20.28
    0.9700000286102295,-9.680000305175781,-0.10999999940395355,2018-10-10 12:54:20.357
    2.0399999618530273,-9.680000305175781,0.27000001072883606,2018-10-10 12:54:20.423

    :return: A tuple containing the standard deviations per accelerometer
        dimension
    """
    accel_data = pd.read_csv(file_path, parse_dates=[3])

    return from_df(accel_data)


def from_df(dataframe):
    """
    :param dataframe: The input data frame which could have an arbitrary
        structure but it is assumed that all columns except a column named
        'timestamp' are numeric.

    :return: A tuple containing the standard deviations of each column except
        the timestamp column
    """
    return tuple(
        [np.std(dataframe[c], ddof=1).item()
         for c in dataframe.columns if not c == 'timestamp'])
