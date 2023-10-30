import torch

from multiscale_operator.transforms.cache import CachedTransform


class AbsolutePositionTransform(CachedTransform):
    def __init__(self):
        super().__init__(check_attributes=["pos"])

    def transform(self, sample_in):
        return sample_in.pos

    def postprocess(self, sample_in, sample_out):
        sample_in.x = torch.cat([sample_in.x, sample_out], dim=-1)
        return sample_in
