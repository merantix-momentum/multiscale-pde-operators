class ChainedTransforms:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample_in):
        for transform in self.transforms:
            sample_in = transform(sample_in)
        return sample_in
