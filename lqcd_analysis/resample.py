import numpy as np
import hashlib

class Bootstrap:
    def __init__(self, data, seed, nresample=None, nensemble=None):
        """
        Args:
            data: array or dict of arrays
            seed: int seed for the random number generator
            nresample: int number of times to sample from original
                data for each bootstrap ensemble
            nensemble: int total number of bootstrap ensembles
        Returns:
            Bootstrap object

        Notes:
        "nconfigs" is the number of configurations in the original dataset.
        Following the notation of DeGrand and DeTar Sec 9.2.3, "nconfigs = N".

        "nresample" denotes the number of times to sample from the original
        data to produce a single bootstrap ensemble. Following the notation
        of DeGrand and DeTar Sec 9.2.3, "nresample = M".

        "nensemble" denotes the total number of bootstrap ensembles to
        consider. Following the notation of DeGrand and DeTar Sec. 9.2.3,
        "nensemble = P"
        """
        self.data = data
        self.is_dict = self._is_dict()
        nconfigs = self._nconfigs()
        if nresample is None:
            nresample = nconfigs
        if nensemble is None:
            nensemble = 4 * nconfigs
        np.random.seed(seed)
        self.seed = seed
        self.nconfigs = nconfigs
        self.nresample = nresample
        self.nensemble = nensemble
        self.draws = np.random.randint(
            low=0,
            high=nconfigs,
            size=(nensemble, nresample)
        )
        self.checksums = self._checksums()

    @property
    def info(self):
        return {
            'seed': self.seed,
            'nconfigs':self.nconfigs,
            'nresample': self.nresample,
            'nensemble': self.nensemble,
        }

    def _is_dict(self):
        attrs = ['keys','values','items']
        return np.all([hasattr(self.data, attr) for attr in attrs])

    def _nconfigs(self):
        if self.is_dict:
            nconfigs = [val.shape[0] for val in self.data.values()]
            return np.unique(nconfigs).item()
        return self.data.shape[0]

    def _sample(self, draw):
        if self.is_dict:
            return {key: value[draw, :] for key, value in self.data.items()}
        return self.data[draw, :]

    def _checksums(self):
        def compute_checksum(arr):
            return hashlib.sha256(arr.tobytes()).hexdigest()
        return np.apply_along_axis(compute_checksum, axis=1, arr=self.draws)

    def __iter__(self):
        for checksum, draw in zip(self.checksums, self.draws):
            yield checksum, self._sample(draw)

    def md5(self, draw):
        """Computes the MD5 checksum for the draw."""
        return hashlib.md5(draw.tobytes()).hexdigest()

    def sha256(self, draw):
        """Computes the SHA256 checksum for the draw."""
        return hashlib.sha256(draw.tobytes()).hexdigest()

