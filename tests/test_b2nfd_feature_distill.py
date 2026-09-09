"""Synthetic NFD checks; no dataset, pretrained download, or final-query evaluation."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation, nfd_weight, load_nfd

S = torch.tensor([[1., .8, .1], [.8, 1., .3], [.1, .3, 1.]])
P = torch.eye(3, 4)
H = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [-1., -1., -1., -1.]])


class Teacher(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.calls = 0
    def forward_features(self, x):
        self.calls += 1
        return self.bn(x) + .3


class TinyNet(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.hidden = nn.Linear(4, 4)
        self.hash_layer = nn.Linear(4, bits)
        self.calls = 0
    def forward_with_features(self, x):
        self.calls += 1
        f = self.hidden(self.bn(x)).tanh() + .4
        return self.hash_layer(f), f
    def forward(self, x):
        return self.forward_with_features(x)[0]


class LossTests(unittest.TestCase):
    def module(self, **kwargs):
        return NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2, **kwargs)

    def test_projection_formula_and_trace(self):
        module = self.module()
        self.assertEqual(module.neighbors.tolist(), [[1, 2], [0, 2], [1, 0]])
        f = torch.tensor([[1., 2., 3., 4.], [4., 2., 1., 3.]], requires_grad=True)
        t = torch.tensor([[2., 1., 4., 3.], [1., 2., 4., 3.]], requires_grad=True)
        y = torch.tensor([0, 2])
        loss = module.feature_loss(f, t, y)
        expected = []
        for i, label in enumerate(y):
            q = module.basis[label]
            a = (.25 * torch.eye(4) + .75 * q @ q.T) / (.25 + .75 * module.ranks[label] / 4)
            torch.testing.assert_close(torch.trace(a), torch.tensor(4.))
            e = F.normalize(f[i], dim=0) - F.normalize(t[i].detach(), dim=0)
            expected.append(e @ a @ e)
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward()
        self.assertGreater(f.grad.norm().item(), 0)
        self.assertIsNone(t.grad)

    def test_plain_and_rank_zero_fallback(self):
        f, t = torch.randn(3, 4), torch.randn(3, 4)
        y = torch.arange(3)
        plain = ((F.normalize(f, dim=1)-F.normalize(t, dim=1))**2).sum(1).mean()
        self.assertTrue(torch.equal(self.module(alpha=0).feature_loss(f,t,y), plain))
        degenerate = NeighborhoodFeatureDistillation(Teacher(), S, P[:1].repeat(3,1), topk=2)
        self.assertEqual(degenerate.ranks.tolist(), [0,0,0])
        self.assertTrue(torch.equal(degenerate.feature_loss(f,t,y), plain))

    def test_teacher_bn_and_gradient_routing(self):
        net, module = TinyNet(4), self.module()
        module.train()
        self.assertFalse(module.teacher.training)
        before = copy.deepcopy(module.teacher.state_dict())
        x = torch.randn(6,4)
        _, f = net.forward_with_features(x)
        module(f,x,torch.arange(6)%3).backward()
        self.assertGreater(net.hidden.weight.grad.norm().item(), 0)
        self.assertIsNone(net.hash_layer.weight.grad)
        self.assertEqual(net.bn.num_batches_tracked.item(), 1)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,module.teacher.state_dict()[key]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.teacher.parameters()))

    def test_validation_and_schedule(self):
        for epoch,value in [(1,0),(20,0),(21,.0005),(40,.01),(300,.01)]:
            self.assertAlmostEqual(nfd_weight(epoch),value)
        for kwargs in [dict(alpha=1),dict(alpha=float('nan'))]:
            with self.assertRaises(ValueError): self.module(**kwargs)
        for kwargs in [dict(maximum=-1),dict(maximum=float('inf')),dict(ramp_end=20)]:
            with self.assertRaises(ValueError): nfd_weight(1,**kwargs)
        m=self.module()
        for f in [torch.zeros(3,4),torch.full((3,4),float('nan'))]:
            with self.assertRaises(ValueError): m.feature_loss(f,P,torch.arange(3))


class IntegrationTests(unittest.TestCase):
    def run_tiny(self, path, enabled=False, weight=.01):
        from train import train_val
        path.mkdir()
        module = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2) if enabled else None
        torch.manual_seed(60)
        x=torch.linspace(-1,1,48).reshape(12,4)
        loader=DataLoader(TensorDataset(x,F.one_hot(torch.arange(12)%3,3).float(),torch.arange(12)),
                          batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(60))
        args=argparse.Namespace(variant='b2nfd' if enabled else 'b1',device=torch.device('cpu'),
             code_length=4,num_classes=3,lr=7e-5,epoch=3,test_map=1,lambd=1e-4,
             nfd_weight=weight,nfd_warmup=1,nfd_ramp_end=2,skip_query=True,
             info='synthetic',run_dir=str(path))
        nets=[]
        def factory(bits):
            net=TinyNet(bits); nets.append(net); return net
        args.net=factory
        val=object(); query=object()
        def evaluate(loader,*unused):
            self.assertIs(loader,val)
            return ({'mAP@ALL':.5,'mAP@100':.6,'mAP@1000':.55},None,None,None,None,None)
        with patch('train._evaluate',side_effect=evaluate),patch('train._save_curves'),patch('builtins.print'):
            train_val(args,H,loader,val,query,None,12,S,nfd=module)
        history=json.loads((path/'training_history.json').read_text())
        self.assertEqual(nets[0].calls,12)
        self.assertEqual(nets[0].bn.num_batches_tracked.item(),12)
        self.assertFalse(json.loads((path/'metrics.json').read_text())['query_evaluated'])
        self.assertTrue((path/'best_model_state.pt').exists())
        return history,nets[0].state_dict(),module

    def test_disabled_exact_pairing_active_loss_and_single_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            baseline,bs,_=self.run_tiny(p/'b1')
            disabled,ds,off=self.run_tiny(p/'off',True,0)
            active,_,on=self.run_tiny(p/'on',True)
            self.assertEqual([r['loss'] for r in baseline],[r['loss'] for r in disabled])
            for key in bs: self.assertTrue(torch.equal(bs[key],ds[key]),key)
            self.assertEqual(off.teacher.calls,0)
            self.assertEqual(on.teacher.calls,8)
            self.assertEqual(active[0]['loss'],baseline[0]['loss'])
            self.assertGreater(active[1]['loss_nfd'],0)
            self.assertIn('sampled_nfd_clean_ratio',active[1])

    def test_resnet_features_preserve_state_and_forward(self):
        import network
        from torchvision import models
        with patch.object(network,'_build_resnet',side_effect=lambda *a,**kw: models.resnet34(weights=None)):
            net=network.ResNet(4).eval()
        before=set(net.state_dict())
        with torch.no_grad():
            x=torch.randn(2,3,32,32)
            output,features=net.forward_with_features(x)
            torch.testing.assert_close(net(x),output,rtol=0,atol=0)
        self.assertEqual(features.shape,(2,512))
        self.assertEqual(set(net.state_dict()),before)


class CacheTeacher(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.feature = nn.Linear(4,512)
        self.classify_layer = nn.Linear(512,classes)
    def forward_features(self,x): return self.feature(x)


class EntryTests(unittest.TestCase):
    def fixture(self,directory,alpha=.75):
        from run import load_config
        from utils.experiment import save_cache,build_cache_metadata
        from utils.reliable_relation import state_sha256,tensor_sha256,relation_config
        p=Path(directory)
        argv=['run.py','--variant','b2nfd','--cpu','--paired-stage-seeds','--skip-query',
              '--num-classes','3','--code-length','4','--nfd-topk','2','--nfd-alpha',str(alpha),
              '--parent-b1-config',str(p/'parent.json'),'--similarity-cache',str(p/'relation.pt'),
              '--center-cache',str(p/'centers.pt'),'--teacher-cache',str(p/'teacher.pt')]
        with patch('sys.argv',argv): args=load_config()
        args.variant='b1';args.split_hash='split';args.run_id='parent';args.output_dir=str(p/'runs')
        state=CacheTeacher(3).state_dict()
        args.classifier_sha256=state_sha256(state)
        args.similarity_content_sha256=tensor_sha256(S)
        tm=build_cache_metadata(args,'classifier');sm=build_cache_metadata(args,'similarity')
        args.similarity_hash=sm['config_hash']
        cm=build_cache_metadata(args,'centers');args.center_content_sha256=tensor_sha256(H)
        save_cache(args.teacher_cache,state,tm)
        save_cache(args.similarity_cache,dict(similarity=S,similarity_sha256=tensor_sha256(S),
                   prototypes=torch.eye(3,512),classifier_sha256=args.classifier_sha256,
                   relation_config=relation_config(args)),sm)
        save_cache(args.center_cache,H,cm)
        parent={k:v for k,v in vars(args).items() if k not in ('net','device')}
        Path(args.parent_b1_config).write_text(json.dumps(parent))
        args.variant='b2nfd';args.git_commit='new-child'
        return args,argv

    def test_frozen_parent_load_teacher_hash_and_rng(self):
        from utils.frozen_b1 import load_frozen_b1
        with tempfile.TemporaryDirectory() as folder:
            args,_=self.fixture(folder)
            load_frozen_b1(args)
            rng=torch.get_rng_state().clone()
            with patch('network.ClassifyNet',CacheTeacher): module=load_nfd(args)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(module.dim,512)
            self.assertEqual(args.nfd_projection_ranks,[2,2,2])
            payload=torch.load(args.teacher_cache,weights_only=False)
            payload['value']['feature.weight'][0,0]+=1
            torch.save(payload,args.teacher_cache)
            with self.assertRaisesRegex(ValueError,'teacher content'): load_nfd(args)

    def test_cli_rejects_invalid_and_main_dispatch_identity(self):
        import run
        ids=[]
        with tempfile.TemporaryDirectory() as folder:
            for mode in [0,.75]:
                root=Path(folder)/str(mode);root.mkdir()
                args,argv=self.fixture(root,mode)
                for extra in [['--nfd-alpha','nan'],['--nfd-alpha','1'],['--nfd-topk','3'],
                              ['--nfd-weight','-1'],['--epoch','1'],['--resume-sbr','x']]:
                    with patch('sys.argv',argv+extra),patch('sys.stderr'),self.assertRaises(SystemExit): run.load_config()
                loader=argparse.Namespace(generator=torch.Generator())
                with patch('run.load_config',return_value=args),patch('network.ClassifyNet',CacheTeacher), \
                     patch('run.load_data',return_value=(loader,None,None,None,None,0,0,0,1)), \
                     patch('run.GenerateSimilarityMatrix') as relation,patch('run.GenerateSemanticHashCenters') as centers, \
                     patch('run.train_val') as train,patch('builtins.print'):
                    run.main()
                relation.assert_not_called();centers.assert_not_called();train.assert_called_once()
                self.assertIsNotNone(train.call_args.kwargs['nfd'])
                manifest=json.loads((Path(args.run_dir)/'config.json').read_text())
                self.assertEqual(manifest['nfd_alpha'],mode)
                self.assertIn('nfd_basis_sha256',manifest)
                self.assertTrue((Path(args.run_dir)/'nfd_projection.pt').exists())
                ids.append(args.run_id)
            self.assertNotEqual(*ids)


if __name__ == '__main__':
    unittest.main()
