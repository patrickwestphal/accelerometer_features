from datetime import datetime


class Window(object):
    def __init__(self, start, end, data):
        assert isinstance(start, datetime)
        self.start = start
        assert isinstance(end, datetime)
        self.end = end
        self.data = data

    def __repr__(self):
        return 'Window from %s to %s with data:\n%s...' % (
            self.start.isoformat(), self.end.isoformat(), str(self.data)[:200])

    def __str__(self):
        return 'Window from %s to %s with data:\n%s' % (
            self.start.isoformat(), self.end.isoformat(), str(self.data)[:200])
