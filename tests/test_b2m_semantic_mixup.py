"""Synthetic B2-M checks: no real data, pretrained model, or retrieval experiment."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from utils.semantic_mixup import (SemanticMixup, backward_mix, frozen_batchnorm,
                                  mix_weight, soft_center_loss)

H = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [-1., -1., -1., -1.]])
S = torch.tensor([[1., .8, -.2], [.8, 1., .3], [-.2, .3, 1.]])


class MixupTests(unittest.TestCase):
    def test_schedule_and_invalid_inputs(self):
        for epoch, value in ((1, 0), (20, 0), (21, .005), (40, .1), (300, .1)):
            self.assertAlmostEqual(mix_weight(epoch), value)
        for kwargs in ({'maximum': -1}, {'maximum': float('nan')}, {'ramp_end': 20}):
            with self.assertRaises(ValueError):
                mix_weight(1, **kwargs)
        for h, s, budget in ((H[[0, 0, 2]], S, 1), (H / 2, S, 1), (H, S, 0)):
            with self.assertRaises(ValueError):
                SemanticMixup(h, s, 60, budget=budget)

    def test_pairing_budget_targets_and_rng_restore(self):
        sampler = SemanticMixup(H, S, 60)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        rng = torch.get_rng_state().clone()
        state = sampler.state_dict()
        a, b, beta = sampler.sample(y)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue((y[a] != y[b]).all())
        self.assertTrue(((beta >= 0) & (beta < .5)).all())
        d = (H[y[a]] != H[y[b]]).sum(1)
        self.assertTrue((beta * d <= .25 + 1e-7).all())
        sampler.load_state_dict(state)
        repeated = sampler.sample(y)
        for first, second in zip((a, b, beta), repeated):
            self.assertTrue(torch.equal(first, second))
        sampler.load_state_dict(state)
        images = torch.arange(24).reshape(6, 4).float()
        mixed, targets = sampler.make_batch(images, y)
        torch.testing.assert_close(mixed, (1-beta[:, None])*images[a] + beta[:, None]*images[b])
        center = (H[y[a]] + 1) / 2
        other = (H[y[b]] + 1) / 2
        torch.testing.assert_close((targets-center).abs().sum(1), beta*d)
        self.assertTrue(torch.equal(targets[center == other], center[center == other]))
        self.assertTrue(torch.equal((targets * 2 - 1).sign(), H[y[a]]))

    def test_top_five_ties_and_plain_no_cap(self):
        h = torch.tensor([[1 if (c >> k) & 1 else -1 for k in range(4)] for c in range(8)]).float()
        semantic = SemanticMixup(h, torch.eye(8), 60)
        classes, p = semantic.class_probabilities(0, list(range(8)))
        self.assertEqual(classes, list(range(1, 8)))
        torch.testing.assert_close(p, torch.tensor([.5/7+.5/5]*5 + [.5/7]*2, dtype=torch.float64))
        plain = SemanticMixup(H, S, 60, 'plain', .00001)
        a, b, beta = plain.sample(torch.arange(3).repeat(20))
        self.assertGreater(beta.max().item(), .25)
        self.assertEqual(plain.diagnostics()['cap_triggered']['mean'], 0)

    def test_single_class_consumes_no_randomness(self):
        sampler = SemanticMixup(H, S, 60)
        state = sampler.generator.get_state().clone()
        mixed, targets = sampler.make_batch(torch.ones(3, 4), torch.zeros(3, dtype=torch.long))
        self.assertEqual(mixed.shape, (0, 4))
        self.assertEqual(targets.shape, (0, 4))
        self.assertTrue(torch.equal(state, sampler.generator.get_state()))

    def test_soft_bce_equivalence_entropy_and_extreme_gradients(self):
        z = torch.linspace(-2, 2, 12).reshape(3, 4).requires_grad_()
        t = torch.linspace(0, 1, 12).reshape(3, 4)
        loss, entropy = soft_center_loss(z, t)
        torch.testing.assert_close(loss, F.binary_cross_entropy((z.tanh()+1)/2, t))
        self.assertGreaterEqual((loss-entropy).item(), -1e-7)
        soft = torch.full((3, 4), .2)
        ideal, expected_entropy = soft_center_loss(torch.logit(soft)/2, soft)
        torch.testing.assert_close(ideal, expected_entropy)
        for value in (-100., 0., 100.):
            x = torch.full((3, 4), value, requires_grad=True)
            result, _ = soft_center_loss(x, t)
            result.backward()
            self.assertTrue(torch.isfinite(result))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_bn_restore_affine_gradients_and_exception(self):
        net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 4))
        net.train()
        stats = net[0].running_mean.clone()
        with frozen_batchnorm(net):
            self.assertFalse(net[0].training)
            self.assertTrue(net.training)
            self.assertTrue(net[1].training)
            net(torch.ones(3, 4)).sum().backward()
        self.assertTrue(net[0].training)
        self.assertTrue(torch.equal(stats, net[0].running_mean))
        self.assertIsNotNone(net[0].weight.grad)
        net[0].eval()
        with self.assertRaises(RuntimeError):
            with frozen_batchnorm(net):
                raise RuntimeError('synthetic')
        self.assertFalse(net[0].training)

    def test_disabled_branch_and_sequential_gradient_equivalence(self):
        sampler = SemanticMixup(H, S, 60)
        net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 4))
        images = torch.linspace(-1, 1, 12).reshape(3, 4)
        y = torch.arange(3)
        initial = copy.deepcopy(net)
        clean = net(images).square().mean()
        clean.backward()
        with patch.object(sampler, 'make_batch', side_effect=AssertionError('must skip')):
            before = copy.deepcopy(net.state_dict())
            grads = [p.grad.clone() for p in net.parameters()]
            result = backward_mix(net, images, y, sampler, 0)
        self.assertEqual(result['loss_mix'], 0)
        for key in before:
            self.assertTrue(torch.equal(before[key], net.state_dict()[key]))
        for old, p in zip(grads, net.parameters()):
            self.assertTrue(torch.equal(old, p.grad))
        state = sampler.state_dict()
        result = backward_mix(net, images, y, sampler, .1, True)
        reference_clean = initial(images).square().mean()
        sampler.load_state_dict(state)
        mixed, targets = sampler.make_batch(images, y)
        with frozen_batchnorm(initial):
            reference_mix, _ = soft_center_loss(initial(mixed), targets)
        (reference_clean + .1*reference_mix).backward()
        for actual, expected in zip(net.parameters(), initial.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        self.assertGreater(result['sampled_weighted_mix_grad_norm'], 0)


class TrainingIntegrationTests(unittest.TestCase):
    def test_tiny_tensor_loop_pairing_single_step_and_saved_outputs(self):
        from train import train_val
        from torch.utils.data import DataLoader, TensorDataset
        images = torch.linspace(-1, 1, 24).reshape(6, 4)
        labels = F.one_hot(torch.arange(3).repeat(2), 3).float()
        dataset = TensorDataset(images, labels, torch.arange(6))
        histories, states = [], []
        with tempfile.TemporaryDirectory() as directory:
            for variant, weight in (("b1", 0.), ("b2m", 0.), ("b2m", .1)):
                path = Path(directory)/f"{variant}_{weight}"
                path.mkdir()
                args = argparse.Namespace(variant=variant, device=torch.device('cpu'),
                    lambd=1e-4, mix_mode='semantic', mix_budget=.25, mix_seed=200063,
                    mix_weight=weight, mix_warmup=1, mix_ramp_end=2,
                    code_length=4, num_classes=3, epoch=3, lr=7e-5, test_map=1,
                    skip_query=True, info=variant, run_dir=str(path))
                nets, optimizers = [], []
                class CountingNet(torch.nn.Sequential):
                    def __init__(self, bit):
                        super().__init__(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, bit))
                        self.calls = 0
                        nets.append(self)
                    def forward(self, x):
                        self.calls += 1
                        return super().forward(x)
                class CountingOptimizer(torch.optim.RMSprop):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        self.steps = 0
                        optimizers.append(self)
                    def step(self, *a, **kw):
                        self.steps += 1
                        return super().step(*a, **kw)
                args.net = CountingNet
                torch.manual_seed(60)
                loader = DataLoader(dataset, batch_size=3, shuffle=False,
                                    generator=torch.Generator().manual_seed(60))
                metrics = {'mAP@ALL': .5, 'mAP@100': .6, 'mAP@1000': .55}
                with patch('train.optim.RMSprop', CountingOptimizer), \
                     patch('train._evaluate', return_value=(metrics, None, None, None, None, None)) as evaluate, \
                     patch('train._save_curves'), patch('builtins.print'):
                    summary = train_val(args, H, loader, None, None, None, 6, S)
                self.assertFalse(summary['query_evaluated'])
                self.assertEqual(evaluate.call_count, 3)  # Validation only.
                self.assertEqual(optimizers[0].steps, 6)
                self.assertEqual(nets[0].calls, 10 if weight else 6)
                self.assertEqual(nets[0][0].num_batches_tracked.item(), 6)
                histories.append(json.loads((path/'training_history.json').read_text()))
                states.append(torch.load(path/'best_model_state.pt', weights_only=True))
                if variant == 'b2m':
                    saved = torch.load(path/'mix_sampler_state.pt', weights_only=True)
                    self.assertEqual(saved['epoch'], 3)
                    diagnostic = json.loads((path/'mix_diagnostics/epoch_003.json').read_text())
                    self.assertEqual(diagnostic['valid_pairs'], 6 if weight else 0)
            self.assertEqual([r['loss'] for r in histories[0]], [r['loss'] for r in histories[1]])
            self.assertEqual(histories[0][0]['loss'], histories[2][0]['loss'])
            self.assertNotEqual(histories[0][1]['loss'], histories[2][1]['loss'])
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]))


class EntryTests(unittest.TestCase):
    def parsed(self, directory, extra=()):
        from run import load_config
        argv = ['run.py', '--variant', 'b2m', '--cpu', '--paired-stage-seeds',
                '--parent-b1-config', str(Path(directory)/'parent.json'),
                '--similarity-cache', str(Path(directory)/'relation.pt'),
                '--center-cache', str(Path(directory)/'centers.pt'),
                '--num-classes', '3', '--code-length', '4', '--skip-query']
        with patch('sys.argv', argv + list(extra)):
            return load_config()

    def test_cli_and_actual_frozen_parent(self):
        from utils.experiment import build_cache_metadata, save_cache
        from utils.frozen_b1 import load_frozen_b1
        from utils.reliable_relation import relation_config, tensor_sha256
        with tempfile.TemporaryDirectory() as directory:
            args = self.parsed(directory)
            self.assertEqual(args.mix_budget, .25)
            self.assertEqual(args.mix_seed, 200063)
            for options in (['--mix-weight', 'nan'], ['--mix-budget', '0'],
                            ['--epoch', '1'], ['--force-recompute'], ['--stage', 'centers']):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    self.parsed(directory, options)
            args.variant = 'b1'
            args.split_hash = 'synthetic-split'
            args.classifier_sha256 = 'synthetic-classifier'
            args.similarity_content_sha256 = tensor_sha256(S)
            relation_meta = build_cache_metadata(args, 'similarity')
            args.similarity_hash = relation_meta['config_hash']
            center_meta = build_cache_metadata(args, 'centers')
            args.center_content_sha256 = tensor_sha256(H)
            args.run_id = 'synthetic-parent'
            save_cache(args.similarity_cache, dict(similarity=S,
                       similarity_sha256=tensor_sha256(S), relation_config=relation_config(args),
                       classifier_sha256=args.classifier_sha256), relation_meta)
            save_cache(args.center_cache, H, center_meta)
            parent = {k:v for k,v in vars(args).items() if k not in ('net', 'device')}
            Path(args.parent_b1_config).write_text(json.dumps(parent), encoding='utf-8')
            args.variant = 'b2m'
            args.git_commit = 'new-implementation'
            similarity, centers = load_frozen_b1(args)
            torch.testing.assert_close(similarity, S)
            torch.testing.assert_close(centers, H)
            parent['split_hash'] = 'wrong'
            Path(args.parent_b1_config).write_text(json.dumps(parent), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_frozen_b1(args)

    def test_dispatch_modes_have_distinct_identity_and_no_regeneration(self):
        import run
        run_ids = []
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('semantic', 'plain'):
                args = self.parsed(directory, ['--mix-mode', mode])
                args.output_dir = directory
                args.split_hash = 'synthetic-split'
                args.parent_b1_run_id = 'synthetic-parent'
                from utils.reliable_relation import tensor_sha256
                args.center_content_sha256 = tensor_sha256(H)
                args.similarity_content_sha256 = 'relation'
                loader = argparse.Namespace(generator=torch.Generator())
                with patch('run.load_config', return_value=args), \
                     patch('run.load_data', return_value=(loader, None, None, None, None, 0, 0, 0, 1)), \
                     patch('utils.frozen_b1.load_frozen_b1', return_value=(S, H)), \
                     patch('run.GenerateSimilarityMatrix') as relation, \
                     patch('run.GenerateSemanticHashCenters') as centers, \
                     patch('run.train_val') as training:
                    run.main()
                    relation.assert_not_called()
                    centers.assert_not_called()
                    training.assert_called_once()
                    torch.testing.assert_close(training.call_args.kwargs['similarity'], S)
                    run_ids.append(args.run_id)
                    with self.assertRaises(FileExistsError):
                        run.main()
            self.assertNotEqual(*run_ids)


if __name__ == '__main__':
    unittest.main()
