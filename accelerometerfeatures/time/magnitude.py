import numpy as np
import pandas as pd


def from_file(file_path):
    """
    Calculates the magnitude of a 3D accelerometer reading where each entry is
    composed of
    - a value for the acceleration in x direction
    - a value for the acceleration in y direction
    - a value for the acceleration in z direction
    - a timestamp

    :param file_path:
    :return: A dataframe containing for each entry the magnitude value and a
    timestamp
    """
    accel_data = pd.read_csv(file_path, parse_dates=[3])

    return from_df(accel_data)


def from_df(accel_dataframe):
    magnitude = np.sqrt(
        accel_dataframe.x**2 + accel_dataframe.y**2 + accel_dataframe.z**2)

    return magnitude.to_frame('magnitude').join(accel_dataframe.timestamp)
