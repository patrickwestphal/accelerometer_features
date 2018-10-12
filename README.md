[![Build Status](https://travis-ci.com/patrickwestphal/accelerometer_features.svg?branch=master)](https://travis-ci.com/patrickwestphal/accelerometer_features)

# Accelerometer Features

This repo contains a collection of features which can be derived from an accelerometer reading.
Those features can be used e.g. by machine learning setups for training an accelerometer-based activity detection.

## Time-based features

The following time-based features are currently implemented:

- Magnitude: The square root of the sum of the squares of the x, y and z dimensions of an accelerometer reading. This gives you mainly the intension of acceleration if you are not interested in the sensor position.
- Mean: Takes the mean(s) of a sensor reading

## Frequency-based features

None implemented, yet.
