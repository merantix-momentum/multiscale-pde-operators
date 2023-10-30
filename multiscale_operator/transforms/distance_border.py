import torch

from multiscale_operator.transforms.cache import CachedTransform


class DistanceBorderTransform(CachedTransform):
    """Transform that adds the distance to the border of the image to the node features.
    Note that this only works for darcy flow."""

    def __init__(self, position_attribute="pos", append_x_attribute="x"):
        super().__init__(check_attributes=[position_attribute])
        self.position_attribute = position_attribute
        self.append_x_attribute = append_x_attribute

    def postprocess(self, sample_in, sample_out):
        prev_x = getattr(sample_in, self.append_x_attribute)
        setattr(sample_in, self.append_x_attribute, torch.cat([prev_x, sample_out["distance_to_border"]], dim=-1))
        return sample_in

    def transform(self, sample):
        sample_pos = getattr(sample, self.position_attribute)

        dis_1 = torch.tensor([1, 1]).to(sample_pos.device) - sample_pos
        dis = torch.stack((sample_pos, dis_1), dim=-1)
        return {"distance_to_border": dis.min(dim=-1).values}
