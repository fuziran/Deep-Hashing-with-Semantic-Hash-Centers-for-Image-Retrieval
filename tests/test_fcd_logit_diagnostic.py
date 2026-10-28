"""Small deterministic checks for direct teacher-logit retrieval diagnostics."""
import unittest

import torch

from scripts.diagnose_fcd_logits import center_codes, paired_metrics


class FCDLogitDiagnosticTests(unittest.TestCase):
    def test_hard_and_soft_centers_use_same_binary_hash_space(self):
        centers = torch.tensor([[1., 1.], [-1., 1.], [1., -1.]])
        logits = torch.tensor([[12., 0., 0.], [0., 12., 0.]])
        expected = centers[:2]
        self.assertTrue(torch.equal(center_codes(logits, centers), expected))
        self.assertTrue(torch.equal(center_codes(logits, centers, 2.), expected))
        with self.assertRaisesRegex(ValueError, "positive"):
            center_codes(logits, centers, 0.)

    def test_paired_report_has_zero_difference_for_identical_teachers(self):
        centers = torch.tensor([[1., 1.], [-1., 1.], [1., -1.]])
        train_labels = torch.tensor([0, 1, 2, 1])
        val_labels = torch.tensor([0, 2])
        train_logits = torch.nn.functional.one_hot(train_labels, 3).float() * 12
        val_logits = torch.nn.functional.one_hot(val_labels, 3).float() * 12
        for temperature in (None, 2.):
            report = paired_metrics(train_logits, val_logits, train_logits, val_logits,
                                    train_labels, val_labels, centers, temperature)
            for metric in report.values():
                self.assertEqual(metric["delta"], 0.)
                self.assertEqual(metric["paired_bootstrap_95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
