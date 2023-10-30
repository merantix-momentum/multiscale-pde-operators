import hashlib

import numpy as np
import torch


class CachedTransform:
    def __init__(self, check_attributes):
        self.cache = TransformsCache(check_attributes=check_attributes)

    def __call__(self, sample_in):
        sample_out = self.cache.query(sample_in)
        if sample_out is None:
            sample_out = self.transform(sample_in)
            self.cache.add(sample_in, sample_out)
        return self.postprocess(sample_in, sample_out)

    def transform(self, sample):
        raise NotImplementedError

    def postprocess(self, sample_in, sample_out):
        raise NotImplementedError


class TransformsCache:
    def __init__(self, check_attributes):
        self.check_attributes = check_attributes
        self.cache = {}

    def _hash(self, sample):
        hash = ""

        for attr in self.check_attributes:
            if not hasattr(sample, attr):
                raise ValueError("Sample does not have attribute %s" % attr)
            value = getattr(sample, attr)
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            if isinstance(value, np.ndarray):
                b = value.view(np.uint8)
                hash += hashlib.sha1(b).hexdigest()
            else:
                raise ValueError(f"Caching of {attr} not implemented")

        return hash

    def query(self, sample_in):
        h = self._hash(sample_in)
        if h in self.cache:
            return self.cache[h]
        else:
            return None

    def add(self, sample_in, sample_out):
        h = self._hash(sample_in)
        self.cache[h] = sample_out
