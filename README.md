[![Build Status](https://travis-ci.com/patrickwestphal/accelerometer_features.svg?branch=master)](https://travis-ci.com/patrickwestphal/accelerometer_features)

# Accelerometer Features

This repo contains a collection of features which can be derived from an accelerometer reading.
Those features can be used e.g. by machine learning setups for training an accelerometer-based activity detection.

## Time-based features

The following time-based features are currently implemented:

- Magnitude: The square root of the sum of the squares of the x, y and z dimensions of an accelerometer reading. This gives you mainly the intension of acceleration if you are not interested in the sensor position.
- Mean: Takes the mean(s) of a sensor reading

Future feature candidates are:

- [Average deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)
- [Skewness](https://en.wikipedia.org/wiki/Skewness#Sample_skewness)
- [Kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Sample_kurtosis)
- [RMS amplitude](https://en.wikipedia.org/wiki/Amplitude#Root_mean_square_amplitude)
- Variance (for each pair; max of pairs, avg of pairs, ...)
- Covariance (for each pair; max of pairs, avg of pairs, ...)
- Time between peaks
- [Binned distribution](http://www.techfak.uni-bielefeld.de/isy-praktikum/WS12SS13/VITAL/media/p74-kwapisz.pdf)

## Frequency-based features

- Fourier transformation

Future feature candidates are:

- [Spectral standard deviation](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Spectral centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
- [Spectral skewness](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Spectral kurtosis](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf) ([also](https://hal.archives-ouvertes.fr/hal-00021302/document))
- [Spectral crest](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Irregularity-K](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Irregularity-J](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Smoothness](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Flatness](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Roll off](https://synrg.csl.illinois.edu/papers/AccelPrint_NDSS14.pdf)
- [Energy](http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=2519&context=sis_research)
- [Entropy](http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=2519&context=sis_research)


Further features from audio signal processing can be found [here](http://docs.twoears.eu/en/latest/afe/available-processors/spectral-features/#jensen2004)
