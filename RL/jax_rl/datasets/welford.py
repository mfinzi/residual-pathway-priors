import numpy as np

class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None):
        self._K = np.float64(0.)
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)

    def add_data(self, x):
        """Add data.
        """
        if x is None:
            return

        x = np.array(x)
        self.n += 1.
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)

        self._Ex += (x - self._K) / self.n
        self._Ex2 += (x - self._K) * (x - self._Ex)
        self._K = self._Ex

    def __call__(self, x):
        self.add_data(x)

    def max(self):
        """Max value for each element in array.
        """
        return self._max

    def min(self):
        """Min value for each element in array.
        """
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.
           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return np.zeros(self.shape)

        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

#    def __add__(self, val):
#        """Add two Welford objects.
#        """
#

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))