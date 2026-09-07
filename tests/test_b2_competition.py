"""Small synthetic correctness checks; no datasets or network training."""
import unittest

import torch
import torch.nn.functional as F

from utils.competition import CenterCompetition, competition_weight


class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.centers = torch.tensor([[1., 1., 1., 1.],
                                     [1., -1., 1., -1.],
                                     [-1., -1., -1., -1.]])
        self.relation = torch.eye(3)

    def test_schedule_boundaries(self):
        for epoch, expected in ((1, 0), (20, 0), (21, .005),
                                (30, .05), (40, .1), (300, .1)):
            self.assertAlmostEqual(competition_weight(epoch, .1), expected)
        with self.assertRaises(ValueError):
            competition_weight(21, .1, 20, 20)

    def test_plain_competition_is_center_cross_entropy(self):
        loss = CenterCompetition(self.centers, self.relation, m0=0, m1=0)
        u = torch.tensor([[.2, -.3, .5, .7], [-.8, .1, -.2, .4]])
        target = torch.tensor([0, 2])
        expected = F.cross_entropy(u @ self.centers.T / 4 / .1, target)
        torch.testing.assert_close(loss(u, target), expected)

    def test_margin_is_reachable_at_target_centers(self):
        relation = torch.ones(3, 3)
        loss = CenterCompetition(self.centers, relation)
        ideal_advantage = 2 * loss.distance / 4
        self.assertTrue((loss.margin <= loss.distance / 4).all())
        self.assertTrue((loss.margin <= ideal_advantage).all())
        self.assertEqual(list(loss.parameters()), [])

    def test_zero_and_saturated_outputs_have_finite_gradients(self):
        loss = CenterCompetition(self.centers, self.relation)
        for value in (0., 1., -1.):
            u = torch.full((3, 4), value, requires_grad=True)
            result = loss(u, torch.arange(3))
            result.backward()
            self.assertTrue(torch.isfinite(result))
            self.assertTrue(torch.isfinite(u.grad).all())

    def test_reject_unreachable_margin_and_invalid_centers(self):
        with self.assertRaises(ValueError):
            CenterCompetition(self.centers, self.relation, m0=.8, m1=.4)
        with self.assertRaises(ValueError):
            CenterCompetition(self.centers * .5, self.relation)


if __name__ == "__main__":
    unittest.main()
