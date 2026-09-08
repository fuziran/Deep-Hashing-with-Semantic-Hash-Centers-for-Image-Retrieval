"""Synthetic SBR validation; no real images, pretrained weights, or query experiment."""
import argparse
import copy
import json
import random

import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from utils.semantic_rank import SemanticRank, binary_code, proxy_distances, rank_weight

H = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [-1., -1., -1., -1.]])
S = torch.tensor([[1., .8, -.2], [.8, 1., .3], [-.2, .3, 1.]])


class RankTests(unittest.TestCase):
    def test_schedule_and_validation(self):
        for epoch, value in ((1, 0), (20, 0), (21, .0025), (40, .05), (300, .05)):
            self.assertAlmostEqual(rank_weight(epoch), value)
        for kwargs in ({'maximum': float('nan')}, {'maximum': -1}, {'ramp_end': 20}):
            with self.assertRaises(ValueError):
                rank_weight(1, **kwargs)
        for kwargs in ({'capacity': 0}, {'max_age': -1}, {'chunk_size': 0}, {'mode': 'bad'}):
            with self.assertRaises(ValueError):
                SemanticRank(H, S, **kwargs)
        with self.assertRaises(ValueError):
            SemanticRank(H[[0, 0, 2]], S)

    def test_queue_last_id_wins_capacity_age_and_exclusion(self):
        ranker = SemanticRank(H, S, capacity=3, max_age=2)
        ranker.enqueue(H, torch.arange(3), torch.tensor([5, 6, 5]), 0)
        self.assertEqual(ranker.ids.tolist(), [6, 5])
        self.assertEqual(ranker.labels.tolist(), [1, 2])
        ranker.enqueue(H[:2], torch.tensor([0, 1]), torch.tensor([6, 7]), 1)
        self.assertEqual(ranker.ids.tolist(), [5, 6, 7])
        ranker.expire(2)
        self.assertEqual(len(ranker.ids), 3)  # Age == max_age remains.
        ranker.expire(3)
        self.assertEqual(ranker.ids.tolist(), [6, 7])
        u = H[:1].clone().requires_grad_()
        loss = ranker.loss(u, torch.tensor([0]), torch.tensor([6]), 3)
        self.assertEqual(loss.item(), 0)  # Own historical record cannot be positive.
        self.assertEqual(ranker.diagnostics()['valid_anchors'], 0)
        ranker.expire(4)
        self.assertEqual(len(ranker.ids), 0)
        with self.assertRaises(ValueError):
            ranker.expire(3)

    def test_class_uniform_mass_and_top_five_ties(self):
        h = torch.tensor([[1 if c >> k & 1 else -1 for k in range(4)] for c in range(8)]).float()
        labels = torch.tensor([1, 1, 1, 2, 3, 4, 5, 6, 7])
        for mode in ('semantic', 'uniform'):
            ranker = SemanticRank(h, torch.eye(8), mode=mode)
            weights, near = ranker.negative_weights(0, labels)
            self.assertEqual(near, [1, 2, 3, 4, 5])
            self.assertAlmostEqual(weights.sum().item(), 1)
            for c in range(1, 8):
                expected = .5/7 + (.5/5 if c <= 5 else 0) if mode == 'semantic' else 1/7
                self.assertAlmostEqual(weights[labels == c].sum().item(), expected)

    def test_proxy_forward_and_analytical_anchor_gradient(self):
        current = torch.tensor([[0., .2, -.5, .8]], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([[.1, -.4, .2, .9]], dtype=torch.float64, requires_grad=True)
        distance, hard, _ = proxy_distances(current, old)
        self.assertTrue(torch.equal(distance, hard))
        self.assertEqual(hard.item(), .5)
        distance.sum().backward()
        torch.testing.assert_close(current.grad, -old.detach()/8)
        self.assertIsNone(old.grad)
        self.assertEqual(binary_code(torch.tensor([0.])).item(), 1)

    def test_loss_matches_explicit_binary_triplets_and_gradient(self):
        old = torch.tensor([[.7,.8,.3,.4], [-.7,-.8,-.3,-.4], [.1,-.2,.3,-.4]], dtype=torch.float64)
        labels, ids = torch.tensor([0,0,1]), torch.tensor([10,11,12])
        current = torch.tensor([[.2,.4,.1,.3]], dtype=torch.float64, requires_grad=True)
        ranker = SemanticRank(H, S, mode='uniform')
        ranker.enqueue(old.requires_grad_(), labels, ids, 0)
        actual = ranker.loss(current, torch.tensor([0]), torch.tensor([1]), 1)
        # Positive distances 0,4 and negative distance 2 -> (0 + 3/4)/2.
        self.assertAlmostEqual(actual.item(), .375)
        actual.backward()
        torch.testing.assert_close(current.grad, (old.detach()[2:3]-old.detach()[1:2])/16)
        self.assertIsNone(old.grad)
        stats = ranker.diagnostics()
        self.assertEqual(stats['strict_errors'], 1)
        self.assertEqual(stats['triplets'], 2)
        self.assertEqual(stats['strict_error_rate'], .5)

    def test_one_bit_margin_zero_and_tie_positive(self):
        ranker = SemanticRank(H, S)
        old = torch.tensor([[.8,.8,.8,.8], [-.4,.4,.4,.4]])
        ranker.enqueue(old, torch.tensor([0,1]), torch.tensor([10,11]), 0)
        u = torch.tensor([[.6,.6,.6,.6]], requires_grad=True)
        loss = ranker.loss(u, torch.tensor([0]), torch.tensor([2]), 1)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(u.grad.abs().sum().item(), 0)
        ranker.enqueue(torch.tensor([[.4,.4,.4,.4]]), torch.tensor([1]), torch.tensor([11]), 1)
        u.grad = None
        loss = ranker.loss(u, torch.tensor([0]), torch.tensor([2]), 2)
        self.assertEqual(loss.item(), .25)
        loss.backward()
        self.assertGreater(u.grad.abs().sum().item(), 0)

    def test_chunks_empty_extreme_and_rng(self):
        generator = torch.Generator().manual_seed(5)
        old = torch.rand(15, 4, generator=generator)*2-1
        labels = torch.arange(15) % 3
        rng = torch.get_rng_state().clone()
        results = []
        for chunk in (1, 64):
            ranker = SemanticRank(H, S, chunk_size=chunk)
            ranker.enqueue(old, labels, torch.arange(15), 0)
            z = torch.tensor([[100., -100., 0., 1.], [-.4,.6,.3,-.1]], requires_grad=True)
            loss = ranker.loss(z.tanh(), torch.tensor([0,1]), torch.tensor([30,31]), 1)
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            results.append((loss.detach(), z.grad))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b)
        ranker = SemanticRank(H, S)
        empty = torch.empty(0,4, requires_grad=True)
        self.assertEqual(ranker.loss(empty, torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long), 0).item(), 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_saved_queue_roundtrip_and_mismatch(self):
        ranker = SemanticRank(H, S)
        ranker.enqueue(H*.8, torch.arange(3), torch.arange(3), 3)
        restored = SemanticRank(H, S)
        restored.load_state_dict(ranker.state_dict())
        self.assertTrue(torch.equal(restored.codes, ranker.codes))
        with self.assertRaises(ValueError):
            SemanticRank(H, S, mode='uniform').load_state_dict(ranker.state_dict())
        state = ranker.state_dict()
        state['ids'][1] = state['ids'][0]
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)


class IntegrationTests(unittest.TestCase):
    def run_tiny(self, path, variant='b2sbr', weight=.05, resume=None, interrupt=False):
        from train import train_val
        from utils.rank_checkpoint import save_checkpoint
        from torch.utils.data import DataLoader, TensorDataset
        path.mkdir()
        images = torch.linspace(-1, 1, 48).reshape(12, 4)
        labels = F.one_hot(torch.arange(12)%3, 3).float()
        class RandomDataset(TensorDataset):
            def __getitem__(self, index):
                image, label, sample_id = super().__getitem__(index)
                jitter = .01 * (torch.rand(()) + np.random.random() + random.random())
                return image + jitter, label, sample_id
        loader = DataLoader(RandomDataset(images, labels, torch.arange(12)), batch_size=3,
                            shuffle=True, generator=torch.Generator().manual_seed(60))
        args = argparse.Namespace(variant=variant, device=torch.device('cpu'), lambd=1e-4,
            rank_capacity=24, rank_max_age=16, rank_chunk_size=2, rank_mode='semantic',
            rank_weight=weight, rank_warmup=1, rank_ramp_end=2, resume_sbr=resume,
            code_length=4, num_classes=3, epoch=3, lr=7e-5, test_map=1,
            skip_query=True, info=variant, run_dir=str(path), run_id='synthetic-run')
        nets, optimizers = [], []
        class Net(torch.nn.Sequential):
            def __init__(self, bit):
                super().__init__(torch.nn.BatchNorm1d(4), torch.nn.Linear(4,bit))
                self.calls=0
                nets.append(self)
            def forward(self, x):
                self.calls+=1
                return super().forward(x)
        class Optimizer(torch.optim.RMSprop):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.steps=0
                optimizers.append(self)
            def step(self, *a, **kw):
                self.steps+=1
                return super().step(*a, **kw)
        args.net=Net
        torch.manual_seed(60)
        np.random.seed(60)
        random.seed(60)
        def save_and_interrupt(*a, **kw):
            save_checkpoint(*a, **kw)
            if interrupt and a[7] == 2:
                raise RuntimeError('synthetic interruption after epoch 2')
        metrics={'mAP@ALL': .5, 'mAP@100': .6, 'mAP@1000': .55}
        with patch('train.optim.RMSprop', Optimizer), patch('train._save_curves'), \
             patch('train._evaluate', return_value=(metrics,None,None,None,None,None)) as evaluate, \
             patch('train.save_checkpoint', side_effect=save_and_interrupt), patch('builtins.print'):
            train_val(args,H,loader,None,None,None,12,S)
        expected_steps=4 if resume else 12
        self.assertEqual(optimizers[0].steps,expected_steps)
        self.assertEqual(nets[0].calls,expected_steps)
        self.assertEqual(evaluate.call_count,1 if resume else 3)
        self.assertEqual(nets[0][0].num_batches_tracked.item(),12)
        return json.loads((path/'training_history.json').read_text()), copy.deepcopy(nets[0].state_dict())

    def test_disabled_pairing_single_forward_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            baseline, bs=self.run_tiny(p/'b1','b1',0)
            disabled, ds=self.run_tiny(p/'off',weight=0)
            active, _=self.run_tiny(p/'on')
            self.assertEqual([r['loss'] for r in baseline],[r['loss'] for r in disabled])
            for key in bs:
                self.assertTrue(torch.equal(bs[key],ds[key]))
            self.assertEqual(baseline[0]['loss'],active[0]['loss'])
            self.assertGreater(active[1]['rank_diagnostics']['valid_anchors'],0)
            self.assertGreater(active[1]['loss_rank'],0)
            self.assertEqual(disabled[-1]['rank_diagnostics']['queue_size'],0)
            self.assertFalse(json.loads((p/'on'/'metrics.json').read_text())['query_evaluated'])
            checkpoint=torch.load(p/'on'/'last_checkpoint.pt',weights_only=False)
            self.assertEqual(checkpoint['global_step'],12)
            self.assertTrue((p/'on'/'best_model_state.pt').exists())

    def test_epoch_resume_matches_uninterrupted_parameters_and_losses(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            full, expected=self.run_tiny(p/'full')
            with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                self.run_tiny(p/'interrupted',interrupt=True)
            resumed, actual=self.run_tiny(p/'resumed',resume=str(p/'interrupted'/'last_checkpoint.pt'))
            self.assertEqual([r['loss'] for r in full],[r['loss'] for r in resumed])
            for key in expected:
                self.assertTrue(torch.equal(expected[key],actual[key]),key)
            state=torch.load(p/'interrupted'/'last_checkpoint.pt',weights_only=False)
            state['run_id']='wrong-source-or-settings'
            torch.save(state,p/'bad.pt')
            with self.assertRaisesRegex(ValueError,'identical source'):
                self.run_tiny(p/'bad',resume=str(p/'bad.pt'))


class EntryTests(unittest.TestCase):
    def parsed(self, directory, extra=()):
        from run import load_config
        argv = ['run.py', '--variant', 'b2sbr', '--cpu', '--paired-stage-seeds',
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
            self.assertEqual(args.rank_capacity, 1024)
            self.assertEqual(args.rank_weight, .05)
            for options in (['--rank-weight', 'nan'], ['--rank-capacity', '0'],
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
            args.variant = 'b2sbr'
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
            for mode in ('semantic', 'uniform'):
                args = self.parsed(directory, ['--rank-mode', mode])
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
