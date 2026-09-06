import argparse
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from losses import center_competition
from train import CSQLoss
from utils.center_cache import load_explicit_centers
from utils.experiment import build_cache_metadata, save_cache
from GenerateSimilarityMatrix import load_explicit_similarity_cache
from run import load_config


class TestCompetition(unittest.TestCase):
    def test_target_alignment_and_negative_competition(self):
        centers = torch.tensor([[1., 1.], [-1., -1.]])
        target = torch.tensor([0])
        good = center_competition(torch.tensor([[1., .8]]), centers, target, .1)
        bad = center_competition(torch.tensor([[-1., -.8]]), centers, target, .1)
        self.assertLess(good, bad)
        near_negative = torch.tensor([[1., 1.], [1., -1.]])
        code = torch.tensor([[1., .1]])
        self.assertGreater(center_competition(code, near_negative, target, .1),
                           center_competition(code, centers, target, .1))

    def test_gradients_and_frozen_centers(self):
        codes = torch.tensor([[.3, -.1], [0., 0.]], dtype=torch.double, requires_grad=True)
        centers = torch.tensor([[1., 1.], [-1., -1.]], dtype=torch.double, requires_grad=True)
        loss = center_competition(codes, centers, torch.tensor([0, 1]), .1)
        loss.backward()
        self.assertTrue(torch.isfinite(codes.grad).all())
        self.assertIsNone(centers.grad)
        x = torch.tensor([[.3, -.1]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: center_competition(z, centers.detach(), torch.tensor([0]), .1), (x,)))

    def test_zero_weight_exactly_matches_b0(self):
        args = argparse.Namespace(device=torch.device('cpu'), method='B0', lambd=1e-4,
                                  competition_weight=0., competition_temperature=.1)
        centers = torch.tensor([[1., 1.], [-1., -1.]])
        criterion = CSQLoss(args, 2, centers)
        output = torch.tensor([[.2, -.3]], requires_grad=True)
        labels = torch.tensor([[1., 0.]])
        baseline = criterion(output, labels, None, args)
        gradient = torch.autograd.grad(baseline, output)[0]
        args.method = 'C1'
        control = criterion(output, labels, None, args)
        self.assertTrue(torch.equal(baseline, control))
        self.assertTrue(torch.equal(gradient, torch.autograd.grad(control, output)[0]))
        args.competition_weight = .1
        self.assertGreater(criterion(output, labels, None, args), control)


class TestFixedCaches(unittest.TestCase):
    def args(self):
        return argparse.Namespace(dataset='toy', seed=60, split_hash='split', num_classes=2,
                                  code_length=4, git_commit='old', method='B0',
                                  mask_strategy='predicted_argmax', similarity_hash='source',
                                  lr=7e-5, classify_epoch=300, resize_size=256, crop_size=224)

    def test_center_reuse_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'centers.pt')
            args = self.args()
            centers = torch.tensor([[1., -1.]] * 4)
            metadata = build_cache_metadata(args, 'centers')
            save_cache(path, centers, metadata)
            args.method, args.git_commit = 'C1', 'new'
            self.assertTrue(torch.equal(load_explicit_centers(args, path), centers))
            args.similarity_hash = 'wrong'
            with self.assertRaises(RuntimeError):
                load_explicit_centers(args, path)
            args.similarity_hash = 'source'
            save_cache(path, torch.zeros_like(centers), metadata)
            with self.assertRaises(RuntimeError):
                load_explicit_centers(args, path)
            save_cache(path, torch.ones_like(centers), metadata)
            with self.assertRaises(RuntimeError):
                load_explicit_centers(args, path)

    def test_similarity_requires_b0(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'similarity.pt')
            args = self.args()
            metadata = build_cache_metadata(args, 'similarity')
            save_cache(path, torch.eye(2), metadata)
            args.method, args.git_commit = 'C1', 'new'
            self.assertTrue(torch.equal(load_explicit_similarity_cache(args, path), torch.eye(2)))
            metadata['method'] = 'B1'
            save_cache(path, torch.eye(2), metadata)
            with self.assertRaises(RuntimeError):
                load_explicit_similarity_cache(args, path)

    def test_cli_requires_frozen_artifacts(self):
        with patch('sys.argv', ['run.py', '--method', 'C1', '--cpu']):
            with self.assertRaises(SystemExit):
                load_config()
        with patch('sys.argv', ['run.py', '--method', 'C1', '--cpu', '--stage', 'train',
                                '--similarity-cache', 's.pt', '--centers-cache', 'h.pt']):
            self.assertEqual(load_config().method, 'C1')


if __name__ == '__main__':
    unittest.main()
