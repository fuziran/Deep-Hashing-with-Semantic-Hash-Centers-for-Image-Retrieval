"""NFD-neighbor fusion is bounded, label-free and distinct from its controls."""
import unittest

import torch

from utils.neighbor_view_fusion import bounded_view_weight, fused_logits


class NeighborViewFusionTests(unittest.TestCase):
    def test_weight_is_equal_for_equal_or_absent_margin_and_bounded(self):
        first = torch.tensor([0., 1., 0., 1.])
        second = torch.tensor([0., 1., 1., 0.])
        weight = bounded_view_weight(first, second)
        self.assertAlmostEqual(weight[0].item(), .5)
        self.assertAlmostEqual(weight[1].item(), .5)
        self.assertTrue(((weight >= .25) & (weight <= .75)).all())
        self.assertLess(weight[2], .5)
        self.assertGreater(weight[3], .5)

    def test_neighbor_criterion_can_differ_from_global_prototype_control(self):
        prototypes = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        neighbors = torch.tensor([[2], [0], [1]], dtype=torch.long)
        first_features = torch.tensor([[.9, .1]])
        second_features = torch.tensor([[.6, .8]])
        first_logits = torch.tensor([[1., -1.]])
        second_logits = torch.tensor([[-2., 1.]])
        result = fused_logits(first_logits, second_logits,
                              first_features, second_features,
                              prototypes, neighbors)
        self.assertAlmostEqual(result["uniform"][0, 0].item(), -.5)
        self.assertGreater(result["global_weight"][0], result["neighbor_weight"][0])
        self.assertGreater(result["global_prototype"][0, 0], 0)
        self.assertLess(result["nfd_neighbor"][0, 0], 0)

    def test_rejects_neighbor_list_containing_anchor(self):
        prototypes = torch.tensor([[1., 0.], [0., 1.]])
        neighbors = torch.tensor([[0], [0]], dtype=torch.long)
        with self.assertRaises(ValueError):
            fused_logits(torch.ones(1, 2), torch.ones(1, 2),
                         torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]),
                         prototypes, neighbors)


if __name__ == "__main__":
    unittest.main()
