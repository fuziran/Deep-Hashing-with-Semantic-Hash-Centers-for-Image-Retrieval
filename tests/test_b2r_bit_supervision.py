"""I2-R invariants and tiny paired training; no dataset or pretrained download."""
import argparse
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train import CSQLoss, train_val
from utils.bit_supervision import BitSupervision, risk_schedule
from utils.reliable_relation import state_sha256


H = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [-1., -1., -1., -1.]])
S = torch.tensor([[1., .8, -.2], [.8, 1., .3], [-.2, .3, 1.]])


def config(**overrides):
    values = dict(variant='b2r', device=torch.device('cpu'), lambd=1e-4,
                  i2r_rho=.25, i2r_stats_start=16, i2r_warmup=20, i2r_ramp_end=40,
                  comp_temperature=.1, comp_m0=.1, comp_m1=.4, comp_weight=.1,
                  comp_warmup=20, comp_ramp_end=40)
    values.update(overrides)
    return argparse.Namespace(**values)


class BitSupervisionTests(unittest.TestCase):
    def test_geometry_and_exhaustive_distance_identity(self):
        module = BitSupervision(H, S)
        torch.testing.assert_close(module.geometry.sum(1), torch.ones(3, dtype=torch.float64))
        self.assertTrue((module.geometry >= 0).all())
        # Independent scalar computation, including negative similarity clipping.
        for y in range(3):
            denominator = sum(1 + max(S[y, c].item(), 0) for c in range(3) if c != y)
            for k in range(4):
                expected = sum((1 + max(S[y, c].item(), 0)) / denominator
                               / int((H[y] != H[c]).sum())
                               for c in range(3) if c != y and H[y, k] != H[c, k])
                self.assertAlmostEqual(module.geometry[y, k].item(), expected)
            for bits in itertools.product((-1., 1.), repeat=4):
                b = torch.tensor(bits)
                for c in range(3):
                    differing = H[y] != H[c]
                    errors = ((b != H[y]) & differing).sum()
                    self.assertEqual(int((b != H[c]).sum() - (b != H[y]).sum()),
                                     int(differing.sum() - 2 * errors))

    def test_observations_sample_weighted_detached_and_frozen(self):
        module = BitSupervision(H, S)
        u = torch.tensor([[0., 0., 0., 0.], [.8, .6, .4, .2], [-.5, .1, .2, -.3],
                          [.1, -.9, .7, -.2]], requires_grad=True)
        y = torch.tensor([0, 0, 1, 2])
        rng = torch.get_rng_state().clone()
        module.observe(u, y, 15)
        self.assertEqual(module.count.sum().item(), 0)
        module.observe(u[:1], y[:1], 16)
        module.observe(u[1:], y[1:], 20)
        module.freeze()
        expected = torch.stack([((1 - H[c] * u.detach()[y == c]) / 2).mean(0) for c in range(3)])
        torch.testing.assert_close(module.deviation.float(), expected)
        self.assertEqual(module.count.tolist(), [2, 1, 1])
        self.assertFalse(module.deviation_sum.requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        before = module.risk.clone()
        module.observe(u, y, 21)
        self.assertTrue(torch.equal(before, module.risk))
        with self.assertRaises(RuntimeError):
            module.observe(u, y, 20)

    def test_weights_bounds_zero_risk_and_diagnostic_hash(self):
        module = BitSupervision(H, S)
        module.observe(torch.zeros_like(H), torch.arange(3), 16)
        module.freeze()
        weights = module.weights(torch.arange(3), .25, H)
        torch.testing.assert_close(weights.mean(1), torch.ones(3))
        self.assertTrue(((weights >= .8) & (weights <= 1.25)).all())
        self.assertGreater(weights.std().item(), 0)
        payload = module.diagnostics(.25)
        self.assertEqual(payload['content_sha256'], state_sha256(payload['tensors']))
        restored = BitSupervision(H, S)
        restored.load_state_dict(module.state_dict())
        self.assertTrue(torch.equal(weights, restored.weights(torch.arange(3), .25, H)))
        perfect = BitSupervision(H, S)
        perfect.observe(H, torch.arange(3), 20)
        perfect.freeze()
        self.assertTrue(torch.equal(perfect.risk, torch.zeros_like(perfect.risk)))
        self.assertTrue(torch.equal(perfect.weights(torch.arange(3), .25, H), torch.ones_like(H)))

    def test_invalid_inputs_and_schedule(self):
        for epoch, expected in ((1, 0), (20, 0), (21, .0125), (40, .25), (300, .25)):
            self.assertAlmostEqual(risk_schedule(epoch), expected)
        for kwargs in ({'maximum': float('nan')}, {'maximum': -.1},
                       {'stats_start': 0}, {'stats_start': 21}, {'ramp_end': 20}):
            with self.assertRaises(ValueError):
                risk_schedule(1, **kwargs)
        for centers, relation in ((H[[0, 0, 2]], S), (H * .5, S),
                                  (H, torch.full_like(S, float('nan')))):
            with self.assertRaises((ValueError, FloatingPointError)):
                BitSupervision(centers, relation)
        module = BitSupervision(H, S)
        with self.assertRaises(RuntimeError):
            module.weights(torch.arange(3), .25, H)
        module.observe(H[:1], torch.tensor([0]), 16)
        with self.assertRaises(ValueError):
            module.freeze()
        with self.assertRaises(ValueError):
            module.observe(torch.full_like(H, float('inf')), torch.arange(3), 16)

    def test_csqloss_exact_b1_loss_gradient_and_rng_when_disabled(self):
        labels = F.one_hot(torch.arange(3), 3).float()
        for epoch, rho in ((1, .25), (16, .25), (20, .25), (21, 0.), (300, 0.)):
            a, b = config(variant='b1'), config(i2r_rho=rho)
            torch.manual_seed(42)
            baseline = CSQLoss(a, 4, H)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            candidate = CSQLoss(b, 4, H, S)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertIsNone(candidate.competition)
            baseline.epoch = candidate.epoch = epoch
            x = torch.linspace(-2, 2, 12).reshape(3, 4).requires_grad_()
            z = x.detach().clone().requires_grad_()
            first, second = baseline(x, labels, None, a), candidate(z, labels, None, b)
            first.backward(); second.backward()
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(x.grad, z.grad))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_active_loss_matches_weighted_bce_and_gradients_finite(self):
        args = config()
        criterion = CSQLoss(args, 4, H, S)
        criterion.bit_supervision.observe(torch.zeros_like(H), torch.arange(3), 20)
        criterion.bit_supervision.freeze()
        criterion.epoch = 40
        labels = torch.eye(3)
        for value in (0., 30., -30.):
            x = torch.full_like(H, value, requires_grad=True)
            loss = criterion(x, labels, None, args)
            u = x.tanh()
            expected = (F.binary_cross_entropy((u + 1) / 2, (H + 1) / 2, reduction='none')
                        * criterion.bit_supervision.weights(torch.arange(3), .25, u)).mean()
            expected = expected + 1e-4 * (u.abs() - 1).square().mean()
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_tiny_paired_training_and_persisted_statistics(self):
        images = torch.linspace(-1, 1, 24).reshape(6, 4)
        labels = F.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
        dataset = TensorDataset(images, labels, torch.arange(6))
        histories, states = [], []
        with tempfile.TemporaryDirectory() as directory:
            for variant, rho in (('b1', 0), ('b2r', 0), ('b2r', .25)):
                path = Path(directory) / f'{variant}_{rho}'
                path.mkdir()
                args = config(variant=variant, i2r_rho=rho, code_length=4, num_classes=3,
                              epoch=42, net=lambda bit: torch.nn.Linear(4, bit),
                              lr=7e-5, test_map=5, skip_query=True, info=variant,
                              run_dir=str(path))
                torch.manual_seed(60)
                loader = DataLoader(dataset, batch_size=4, shuffle=True,
                                    generator=torch.Generator().manual_seed(60))
                metrics = {'mAP@ALL': .5, 'mAP@100': .6, 'mAP@1000': .55}
                with patch('train._evaluate', return_value=(metrics, None, None, None, None, None)), \
                     patch('train._save_curves'), patch('builtins.print'):
                    summary = train_val(args, H, loader, None, None, None, 6, S)
                self.assertFalse(summary['query_evaluated'])
                history = json.loads((path / 'training_history.json').read_text())
                histories.append(history)
                states.append(torch.load(path / 'best_model_state.pt', weights_only=True))
                if variant == 'b2r':
                    payload = torch.load(path / 'i2r_statistics.pt', weights_only=True)
                    manifest = json.loads((path / 'config.json').read_text())
                    self.assertEqual(payload['tensors']['count'].tolist(), [10, 10, 10])
                    self.assertEqual(manifest['i2r_statistics_sha256'], payload['content_sha256'])
                    self.assertEqual(payload['content_sha256'], state_sha256(payload['tensors']))
            self.assertEqual([r['loss'] for r in histories[0]], [r['loss'] for r in histories[1]])
            self.assertEqual([r['loss'] for r in histories[0][:20]], [r['loss'] for r in histories[2][:20]])
            self.assertNotEqual(histories[0][20]['loss'], histories[2][20]['loss'])
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]))


class EntryAndParentTests(unittest.TestCase):
    def parsed(self, directory, extra=()):
        from run import load_config
        argv = ['run.py', '--variant', 'b2r', '--cpu', '--paired-stage-seeds',
                '--parent-b1-config', str(Path(directory) / 'parent.json'),
                '--similarity-cache', str(Path(directory) / 'relation.pt'),
                '--center-cache', str(Path(directory) / 'centers.pt'),
                '--num-classes', '3', '--code-length', '4', '--skip-query']
        with patch('sys.argv', argv + list(extra)):
            return load_config()

    def test_cli_defaults_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.parsed(directory)
            self.assertEqual(args.i2r_rho, .25)
            self.assertEqual(args.info, '[SHC-B2R]')
            self.assertIn('i2r', args.i2r_version)
            for options in (['--epoch', '1'], ['--i2r-rho', 'nan'],
                            ['--i2r-stats-start', '21'], ['--force-recompute'],
                            ['--stage', 'similarity']):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    self.parsed(directory, options)

    def test_b2r_accepts_only_matching_frozen_b1_artifacts(self):
        from utils.experiment import build_cache_metadata, save_cache
        from utils.frozen_b1 import load_frozen_b1
        from utils.reliable_relation import relation_config, tensor_sha256
        with tempfile.TemporaryDirectory() as directory:
            args = self.parsed(directory)
            args.variant = 'b1'
            args.split_hash = 'synthetic-split'
            args.classifier_sha256 = 'synthetic-classifier'
            args.similarity_content_sha256 = tensor_sha256(S)
            relation_meta = build_cache_metadata(args, 'similarity')
            args.similarity_hash = relation_meta['config_hash']
            center_meta = build_cache_metadata(args, 'centers')
            args.center_content_sha256 = tensor_sha256(H)
            args.run_id = 'synthetic-b1-parent'
            save_cache(args.similarity_cache, dict(similarity=S,
                       similarity_sha256=tensor_sha256(S), relation_config=relation_config(args),
                       classifier_sha256=args.classifier_sha256), relation_meta)
            save_cache(args.center_cache, H, center_meta)
            parent = {k: v for k, v in vars(args).items() if k not in ('net', 'device')}
            Path(args.parent_b1_config).write_text(json.dumps(parent), encoding='utf-8')
            args.variant = 'b2r'
            args.git_commit = 'new-implementation-commit'
            similarity, centers = load_frozen_b1(args)
            self.assertTrue(torch.equal(similarity, S))
            self.assertTrue(torch.equal(centers, H))
            self.assertEqual(args.parent_b1_run_id, 'synthetic-b1-parent')
            for field, changed in (('epoch', 301), ('split_hash', 'wrong-split'),
                                   ('center_content_sha256', 'wrong-digest'),
                                   ('similarity_content_sha256', 'wrong-digest')):
                modified = dict(parent, **{field: changed})
                Path(args.parent_b1_config).write_text(json.dumps(modified), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_frozen_b1(args)


    def test_main_routes_to_frozen_parent_and_separates_run_ids(self):
        import run
        with tempfile.TemporaryDirectory() as directory:
            run_ids = []
            for rho in (.25, 0.):
                args = self.parsed(directory, ['--i2r-rho', str(rho), '--output-dir', directory])
                args.split_hash = 'synthetic-split'
                def parent_loader(current):
                    current.center_content_sha256 = 'synthetic-center'
                    current.similarity_content_sha256 = 'synthetic-relation'
                    current.parent_b1_run_id = 'synthetic-parent'
                    return S, H
                loader = argparse.Namespace(generator=torch.Generator())
                loaded = (loader, None, None, None, None, 0, 0, 0, 6)
                with patch('run.load_config', return_value=args), \
                     patch('run.load_data', return_value=loaded), \
                     patch('utils.frozen_b1.load_frozen_b1', side_effect=parent_loader), \
                     patch('run.GenerateSimilarityMatrix') as generate_relation, \
                     patch('run.GenerateSemanticHashCenters') as generate_centers, \
                     patch('run.train_val') as training:
                    run.main()
                    generate_relation.assert_not_called()
                    generate_centers.assert_not_called()
                    training.assert_called_once()
                    self.assertTrue(torch.equal(training.call_args.kwargs['similarity'], S))
                    manifest = json.loads((Path(args.run_dir) / 'config.json').read_text())
                    self.assertEqual(manifest['i2r_rho'], rho)
                    self.assertEqual(manifest['parent_b1_run_id'], 'synthetic-parent')
                    run_ids.append(args.run_id)
                    with self.assertRaises(FileExistsError):
                        run.main()
            self.assertNotEqual(*run_ids)


if __name__ == '__main__':
    unittest.main()
