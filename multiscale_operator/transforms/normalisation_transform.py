class NormaliseTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample.x = (sample.x - self.mean) / self.std
        return sample
