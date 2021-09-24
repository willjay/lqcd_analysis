from operator import xor
import numpy as np

class Variation:

    def __init__(self, times, nstates):
        self.times = times
        self.nstates = nstates
        self._Times = type(self.times)
        self._Nstates = type(self.nstates)

    def variation(self, **kwargs):
        """
        Creates a updated version of the base case consisting of Times and Nstates.
        """
        times = self.times._asdict()
        nstates = self.nstates._asdict()

        for key in kwargs:
            if not xor(key in times, key in nstates):
                raise ValueError(f"{key} should appear exactly once in 'times' or 'nstates'.")

        for key, value in kwargs.items():
            if key in times:
                times[key] = value
                continue
            if key in nstates:
                nstates[key] = value
                continue

        times = self._Times(**times)
        nstates = self._Nstates(**nstates)
        return times, nstates

class TwoPointVariations(Variation):

    def __call__(self, tmins=None, n_decays=None):
        print("Base case")
        yield (self.times, self.nstates)

        for key, value in self.grid_choice(tmins=tmins, n_decays=n_decays):
            print("Running", key, value)
            times, nstates = self.variation(**{key: value})

            yield (times, nstates)

    def grid_choice(self, tmins=None, n_decays=None):
        """
        Generator for creating variations on a given base_case of Times and Nstates,
        varying one analysis option at a time.
        """

        if tmins is None:
            # default: tmin (do next 3)
            tmins = self.times.tmin + np.arange(1, 4)

        if n_decays is None:
            # default: n (do next 3)
            n_decays = self.nstates.n + np.arange(1, 4)

        for tmin in tmins:
            yield ('tmin', tmin)

        for n_decay in n_decays:
            yield ('n', n_decay)
