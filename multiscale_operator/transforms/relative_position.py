import torch

from multiscale_operator.transforms.cache import CachedTransform


class RelativePositionTransform(CachedTransform):
    def __init__(self, position_attribute="pos", edge_index_attribute="edge_index", edge_attr_attribute="edge_attr"):
        super().__init__(check_attributes=[edge_index_attribute, position_attribute])
        self.position_attribute = position_attribute
        self.edge_index_attribute = edge_index_attribute
        self.edge_attr_attribute = edge_attr_attribute

    def postprocess(self, sample_in, sample_out):
        setattr(sample_in, self.edge_attr_attribute, sample_out["edge_attr"])
        return sample_in

    def transform(self, sample):
        sample_ei = getattr(sample, self.edge_index_attribute)
        sample_pos = getattr(sample, self.position_attribute)

        rel_pos = sample_pos[sample_ei[0]] - sample_pos[sample_ei[1]]
        # compute relative distances
        rel_dist = torch.norm(rel_pos, dim=1, keepdim=True)
        return {"edge_attr": rel_dist}
