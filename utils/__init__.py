import itertools


def pairwise_iterator(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
