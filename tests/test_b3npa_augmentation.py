"""Synthetic I3 formula, training isolation, exact disabled reduction and CLI checks."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tests.test_b2nfd_feature_distill import Teacher, TinyNet, S, P, H, CacheTeacher
from tests import test_b2nfd_feature_distill as nfd_tests
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation
from utils.neighborhood_protected_augmentation import (
    ClassCovariance, NeighborhoodProtectedAugmentation, prepare_covariances,
    prepare_npa, npa_weight,
)


class FormulaTests(unittest.TestCase):
    def test_streaming_covariance_matches_direct_and_shrinks_to_within_class_pool(self):
        x = torch.arange(60,dtype=torch.float64).reshape(12,5).sin()
        y = torch.arange(12)%3
        moments = ClassCovariance(3,5)
        for start,end in [(0,2),(2,7),(7,12)]: moments.update(x[start:end],y[start:end])
        direct = torch.stack([torch.cov(x[y==c].T) for c in range(3)])
        torch.testing.assert_close(moments.finish(0),direct)
        torch.testing.assert_close(moments.finish(1),direct.mean(0).expand_as(direct))
        torch.testing.assert_close(moments.finish(.2),.8*direct+.2*direct.mean(0))

    def test_projection_psd_trace_and_equal_energy_controls(self):
        raw = torch.tensor([[[2.,1.,0.],[1.,3.,0.],[0.,0.,4.]]])
        q = torch.tensor([[[1.],[0.],[0.]]])
        for mode in ['protected','covariance','isotropic']:
            c,retained = prepare_covariances(raw,q,mode)
            torch.testing.assert_close(c[0].trace(),torch.tensor(1.))
            self.assertGreaterEqual(torch.linalg.eigvalsh(c[0]).min(),-1e-6)
            if mode=='protected':
                torch.testing.assert_close(c[0]@q[0],torch.zeros(3,1))
                self.assertAlmostEqual(retained[0],7/9)
            if mode=='isotropic': torch.testing.assert_close(c[0],torch.eye(3)/3)
        with self.assertRaises(ValueError): prepare_covariances(-raw,q)
        with self.assertRaises(ValueError): prepare_covariances(raw,torch.eye(3)[None])

    def test_analytic_upper_bound_and_gradient_routing(self):
        torch.manual_seed(72)
        module = NeighborhoodProtectedAugmentation(torch.eye(4)[None].repeat(3,1,1)/4,.2)
        net = TinyNet(4)
        _,features = net.forward_with_features(torch.randn(6,4))
        output = net.hash_layer(features)
        y = torch.arange(6)%3
        increment = module(output,features,y,net.hash_layer.weight,H)
        variance = (.2**2 * features.detach().square().sum(1,keepdim=True)
                    * net.hash_layer.weight.square().sum(1)[None]/4)
        clean = F.softplus(-2*H[y]*output)
        expected = (F.softplus(-2*H[y]*output+2*variance)-clean).mean()
        torch.testing.assert_close(increment,expected,atol=1e-7,rtol=1e-5)
        # Deterministic integration in one dimension verifies the Gaussian
        # expectation bound independently from the implementation's expression.
        nodes,weights = np.polynomial.hermite.hermgauss(60)
        samples = output.detach().double()[None] + torch.from_numpy(nodes)[:,None,None]*(2*variance.detach().double()).sqrt()[None]
        expectation = (F.softplus(-2*H[y][None]*samples)*torch.from_numpy(weights)[:,None,None]).sum(0)/np.sqrt(np.pi)
        upper = F.softplus(-2*H[y]*output.detach()+2*variance.detach())
        self.assertTrue((expectation <= upper.double()+1e-6).all())
        increment.backward()
        self.assertGreater(net.hash_layer.weight.grad.norm(),0)
        self.assertGreater(net.hidden.weight.grad.norm(),0)
        self.assertFalse(module.covariance.requires_grad)

    def test_schedule_and_invalid_values(self):
        for e,v in [(1,0),(40,0),(41,.05),(60,1),(300,1)]: self.assertAlmostEqual(npa_weight(e),v)
        for kw in [dict(maximum=2),dict(maximum=float('nan')),dict(ramp_end=40)]:
            with self.assertRaises(ValueError): npa_weight(1,**kw)
        with self.assertRaises(ValueError): NeighborhoodProtectedAugmentation(torch.eye(4)[None],.1)
        with self.assertRaises(ValueError): ClassCovariance(3,4).finish()

    def test_zero_rho_extreme_logits_and_bit_mean(self):
        c=torch.eye(4)[None].repeat(3,1,1)/4
        f=torch.ones(2,4,requires_grad=True); w=torch.ones(2,4,requires_grad=True)
        z=torch.tensor([[1000.,-1000.],[-1000.,1000.]],requires_grad=True)
        y=torch.tensor([0,1]); h=torch.tensor([[1.,1.],[-1.,-1.],[1.,-1.]])
        self.assertEqual(NeighborhoodProtectedAugmentation(c,0)(z,f,y,w,h).item(),0)
        module=NeighborhoodProtectedAugmentation(c,.05)
        loss=module(z,f,y,w,h)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue(torch.isfinite(w.grad).all())
        repeated=module(z.detach().repeat(1,2),f,y,w.detach().repeat(2,1),h.repeat(1,2))
        torch.testing.assert_close(loss,repeated)


class PreparationTests(unittest.TestCase):
    def fixture(self):
        x=torch.linspace(-1,1,48).reshape(12,4)
        dataset=TensorDataset(x,F.one_hot(torch.arange(12)%3,3).float(),torch.arange(12))
        dataset.indices=np.arange(12)
        return DataLoader(dataset,batch_size=4,generator=torch.Generator().manual_seed(60)),NeighborhoodFeatureDistillation(Teacher(),S,P,topk=2)

    def args(self,**overrides):
        values=dict(npa_weight=1.,npa_rho=.05,npa_mode='protected',npa_shrinkage=.2,
                    num_classes=3,device=torch.device('cpu'),nfd_teacher_sha256='teacher',
                    nfd_basis_sha256='basis',split_hash='split',seed=60,non_deterministic=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_training_only_preparation_rng_teacher_and_identity(self):
        loader,nfd=self.fixture()
        state=copy.deepcopy(nfd.teacher.state_dict());rng=torch.get_rng_state().clone()
        module=prepare_npa(self.args(),nfd,loader,loader)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(nfd.teacher.calls,6)
        for key in state: self.assertTrue(torch.equal(state[key],nfd.teacher.state_dict()[key]))
        self.assertEqual(module.artifact['identity']['counts'],[4,4,4])
        for c in range(3):
            torch.testing.assert_close(module.covariance[c]@nfd.basis[c],torch.zeros(4,2),atol=1e-6,rtol=0)
        self.assertTrue(all(p.grad is None for p in nfd.teacher.parameters()))

    def test_disabled_does_not_access_loader_and_mismatched_ids_rejected(self):
        self.assertIsNone(prepare_npa(self.args(npa_weight=0),None,None,None))
        loader,nfd=self.fixture();other,_=self.fixture();other.dataset.indices=np.arange(12)+1
        with self.assertRaisesRegex(ValueError,'training image IDs'): prepare_npa(self.args(),nfd,other,loader)


class TrainingTests(unittest.TestCase):
    def run_tiny(self,path,variant='b2nfd',enabled=False,weight=1.,rho=.05,nfd_strength=.01):
        from train import train_val
        path.mkdir()
        nfd=NeighborhoodFeatureDistillation(Teacher(),S,P,topk=2)
        npa=NeighborhoodProtectedAugmentation(torch.eye(4)[None].repeat(3,1,1)/4,rho) if enabled else None
        torch.manual_seed(60)
        x=torch.linspace(-1,1,48).reshape(12,4)
        loader=DataLoader(TensorDataset(x,F.one_hot(torch.arange(12)%3,3).float(),torch.arange(12)),
                          batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(60))
        args=argparse.Namespace(variant=variant,device=torch.device('cpu'),code_length=4,num_classes=3,
             lr=7e-5,epoch=3,test_map=1,lambd=1e-4,nfd_weight=nfd_strength,nfd_warmup=1,nfd_ramp_end=2,
             npa_weight=weight,npa_rho=rho,npa_warmup=1,npa_ramp_end=2,skip_query=True,info='tiny',run_dir=str(path))
        nets=[]
        def factory(bits):
            net=TinyNet(bits);nets.append(net);return net
        args.net=factory;val=object()
        def evaluate(loader,*unused):
            self.assertIs(loader,val)
            return ({'mAP@ALL':.5,'mAP@100':.6,'mAP@1000':.55},None,None,None,None,None)
        with patch('train._evaluate',side_effect=evaluate),patch('train._save_curves'),patch('builtins.print'):
            train_val(args,H,loader,val,object(),None,12,S,nfd=nfd,npa=npa)
        self.assertEqual(nets[0].calls,12);self.assertEqual(nets[0].bn.num_batches_tracked.item(),12)
        self.assertFalse(json.loads((path/'metrics.json').read_text())['query_evaluated'])
        self.assertTrue(all('covariance' not in key and 'teacher' not in key for key in nets[0].state_dict()))
        return json.loads((path/'training_history.json').read_text()),nets[0].state_dict(),nfd

    def test_disabled_exact_nfd_pairing_and_active_single_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            baseline,bs,_=self.run_tiny(p/'base')
            disabled,ds,_=self.run_tiny(p/'off','b3npa',weight=0)
            zero,zs,_=self.run_tiny(p/'rho0','b3npa',rho=0)
            active,_,_=self.run_tiny(p/'on','b3npa',True)
            self.assertEqual([r['loss'] for r in baseline],[r['loss'] for r in disabled])
            self.assertEqual([r['loss'] for r in baseline],[r['loss'] for r in zero])
            for key in bs:
                self.assertTrue(torch.equal(bs[key],ds[key]),key)
                self.assertTrue(torch.equal(bs[key],zs[key]),key)
            self.assertEqual(active[0]['loss'],baseline[0]['loss'])
            self.assertGreater(active[1]['loss_npa'],0)
            self.assertIn('sampled_npa_clean_ratio',active[1])

    def test_npa_active_without_nfd_weight(self):
        with tempfile.TemporaryDirectory() as folder:
            history,_,nfd=self.run_tiny(Path(folder)/'on','b3npa',True,nfd_strength=0)
            self.assertGreater(history[1]['loss_npa'],0)
            self.assertEqual(nfd.teacher.calls,0)


class EntryIntegrationTests(unittest.TestCase):
    def test_cli_dispatch_frozen_artifacts_and_distinct_modes(self):
        import run
        identities=[]
        with tempfile.TemporaryDirectory() as folder:
            for mode in ['protected','covariance','isotropic']:
                root=Path(folder)/mode;root.mkdir()
                args,argv=nfd_tests.EntryTests().fixture(root)
                argv[argv.index('b2nfd')]='b3npa'
                argv+=['--npa-mode',mode]
                with patch('sys.argv',argv): child=run.load_config()
                for key in ['split_hash','output_dir']:
                    setattr(child,key,getattr(args,key))
                loader,nfd=PreparationTests().fixture()
                # Use real 512-dimensional teacher preparation with actual parent
                # cache checks; no real images or external weights are needed.
                with patch('run.load_config',return_value=child),patch('network.ClassifyNet',CacheTeacher), \
                     patch('run.load_data',return_value=(loader,loader,None,None,None,0,0,0,12)), \
                     patch('run.GenerateSimilarityMatrix') as relation,patch('run.GenerateSemanticHashCenters') as centers, \
                     patch('run.train_val') as train,patch('builtins.print'):
                    run.main()
                relation.assert_not_called();centers.assert_not_called()
                self.assertIsNotNone(train.call_args.kwargs['npa'])
                manifest=json.loads((Path(child.run_dir)/'config.json').read_text())
                self.assertEqual(manifest['npa_mode'],mode)
                self.assertIn('nfd_alpha',manifest);self.assertIn('npa_covariance_sha256',manifest)
                self.assertTrue((Path(child.run_dir)/'npa_statistics.pt').exists())
                identities.append(child.run_id)
                for extra in [['--npa-rho','nan'],['--npa-shrinkage','2'],['--npa-weight','2'],['--npa-ramp-end','301']]:
                    with patch('sys.argv',argv+extra),patch('sys.stderr'),self.assertRaises(SystemExit): run.load_config()
            self.assertEqual(len(set(identities)),3)


if __name__=='__main__': unittest.main()
