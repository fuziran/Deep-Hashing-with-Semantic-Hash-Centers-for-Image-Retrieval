"""Synthetic foundation-distillation checks: no downloads or real query data."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset

from tests.test_b2nfd_feature_distill import S,P,H,TinyNet,Teacher,CacheTeacher
from tests import test_b2nfd_feature_distill as nfd_tests
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation
from utils.foundation_distill import (FoundationDistillation,decoupled_loss,ramp,load_fcd,
                                      CACHE_VERSION,image_subset_sha256)
from utils.reliable_relation import tensor_sha256


def dataset_fixture():
    x=torch.linspace(-1,1,48).reshape(12,4)
    y=torch.arange(12)%3
    dataset=TensorDataset(x,F.one_hot(y,3).float(),torch.arange(12))
    dataset.indices=np.arange(12);dataset.images=x.numpy();dataset.labels=y.numpy()
    return dataset


def cache_fixture(path,args,dataset,passed=True):
    ids=torch.tensor(np.sort(dataset.indices));labels=torch.tensor(dataset.labels[ids])
    logits=F.one_hot(labels,args.num_classes).float()*4
    identity=dict(version=CACHE_VERSION,dataset=args.dataset,seed=args.seed,split_hash=args.split_hash,
                  num_classes=args.num_classes,classifier_sha256=args.classifier_sha256,
                  resize_size=args.resize_size,crop_size=args.crop_size,
                  training_images_sha256=image_subset_sha256(dataset),foundation_weights_sha256='synthetic-weights')
    for name,value in [('ids',ids),('labels',labels),('logits',logits)]: identity[name+'_sha256']=tensor_sha256(value)
    payload=dict(identity=identity,ids=ids,labels=labels,logits=logits,teacher_check=dict(passed=passed))
    torch.save(payload,path);return payload


class FormulaTests(unittest.TestCase):
    def module(self,mode='neighbor',gamma=1.):
        ids=torch.arange(12);y=ids%3
        return FoundationDistillation(H,S,ids,F.one_hot(y,3).float()*4,y,mode,2.,gamma,2)

    def test_dkd_matches_independent_kl_decomposition_and_gradient(self):
        torch.manual_seed(33)
        s=torch.randn(5,4,requires_grad=True);t=torch.randn(5,4,requires_grad=True);y=torch.arange(5)%4
        a,b=decoupled_loss(s,t,y)
        ps,pt=(s/2).softmax(1),(t.detach()/2).softmax(1)
        reference_a=[];reference_b=[]
        for i,target in enumerate(y):
            mask=torch.arange(4)!=target
            qs=torch.stack([ps[i,target],ps[i,mask].sum()])
            qt=torch.stack([pt[i,target],pt[i,mask].sum()])
            reference_a.append((qt*(qt.log()-qs.log())).sum()*4)
            qs=ps[i,mask]/ps[i,mask].sum();qt=pt[i,mask]/pt[i,mask].sum()
            reference_b.append((qt*(qt.log()-qs.log())).sum()*4)
        torch.testing.assert_close(a,torch.stack(reference_a).mean())
        torch.testing.assert_close(b,torch.stack(reference_b).mean())
        (a+b).backward();self.assertIsNone(t.grad);self.assertGreater(s.grad.norm(),0)

    def test_controls_weights_identity_and_extreme_values(self):
        s=torch.tensor([[1000.,-1000.,0.],[-1000.,1000.,0.]],requires_grad=True)
        t=-s.detach();y=torch.tensor([0,1])
        plain=decoupled_loss(s,t,y);weighted=decoupled_loss(s,t,y,weights=torch.ones_like(s))
        for x,z in zip(plain,weighted): torch.testing.assert_close(x,z,atol=0,rtol=0)
        sum(plain).backward();self.assertTrue(torch.isfinite(s.grad).all())
        logits=torch.randn(3,4)
        a=self.module('plain');b=self.module('neighbor',0)
        torch.testing.assert_close(a(logits.tanh(),torch.arange(3),torch.arange(3)),
                                   b(logits.tanh(),torch.arange(3),torch.arange(3)),atol=0,rtol=0)
        weights=self.module().negative_weights
        self.assertTrue(((weights>=1)&(weights<=2)).all())
        torch.testing.assert_close(weights.diag(),torch.ones(3))

    def test_rng_head_gradient_teacher_detachment_and_id_guards(self):
        rng=torch.get_rng_state().clone();module=self.module()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        codes=torch.randn(3,4,requires_grad=True)
        module(codes.tanh(),torch.arange(3),torch.arange(3)).backward()
        self.assertGreater(codes.grad.norm(),0);self.assertGreater(module.head.weight.grad.norm(),0)
        self.assertFalse(module.teacher_logits.requires_grad)
        with self.assertRaisesRegex(ValueError,'outside'): module(codes,torch.arange(3),torch.tensor([0,1,99]))
        with self.assertRaisesRegex(ValueError,'label'): module(codes,torch.tensor([0,0,0]),torch.arange(3))
        module=self.module('ce');module(codes,torch.arange(3),torch.arange(3))
        self.assertEqual(module.components['fcd_nckd'],0)

    def test_options_and_schedule(self):
        for e,v in [(1,0),(40,0),(41,.0025),(60,.05),(300,.05)]: self.assertAlmostEqual(ramp(e),v)
        with self.assertRaises(ValueError): ramp(1,float('nan'))
        with self.assertRaises(ValueError): decoupled_loss(torch.ones(2,3),torch.ones(2,3),torch.tensor([0,1]),0)


class CacheTests(unittest.TestCase):
    def args(self,path):
        return argparse.Namespace(fcd_weight=.05,fcd_cache=str(path),dataset='cifar-100-new-seg',seed=60,
           split_hash='split',num_classes=3,classifier_sha256='old',resize_size=256,crop_size=224,
           code_length=4,fcd_mode='neighbor',fcd_temperature=2.,fcd_gamma=1.,fcd_topk=2,device=torch.device('cpu'))

    def test_content_images_labels_and_information_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.pt';args=self.args(path);data=dataset_fixture()
            payload=cache_fixture(path,args,data)
            self.assertIsNotNone(load_fcd(args,H,S,data))
            payload['logits'][0,0]+=1;torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'content'):load_fcd(args,H,S,data)
            cache_fixture(path,args,data,False)
            with self.assertRaisesRegex(ValueError,'information'):load_fcd(args,H,S,data)
            cache_fixture(path,args,data);data.images[0,0]+=.5
            with self.assertRaisesRegex(ValueError,'images'):load_fcd(args,H,S,data)
            args.fcd_weight=0
            self.assertIsNone(load_fcd(args,None,None,None))


class TrainingTests(unittest.TestCase):
    def run_tiny(self,path,mode=None,weight=.05):
        from train import train_val
        path.mkdir();data=dataset_fixture()
        fcd=FormulaTests().module(mode) if mode and weight>0 else None
        nfd=NeighborhoodFeatureDistillation(Teacher(),S,P,topk=2)
        torch.manual_seed(60)
        loader=DataLoader(data,batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(60))
        args=argparse.Namespace(variant='b3fcd' if mode else 'b2nfd',device=torch.device('cpu'),
              code_length=4,num_classes=3,lr=7e-5,epoch=3,test_map=1,lambd=1e-4,
              nfd_weight=.01,nfd_warmup=1,nfd_ramp_end=2,fcd_weight=weight,fcd_warmup=1,fcd_ramp_end=2,
              skip_query=True,info='synthetic',run_dir=str(path))
        nets=[]
        def factory(bits):
            net=TinyNet(bits);nets.append(net);return net
        args.net=factory;val=object()
        def evaluate(loader,*unused):
            self.assertIs(loader,val)
            return ({'mAP@ALL':.5,'mAP@100':.6,'mAP@1000':.55},None,None,None,None,None)
        with patch('train._evaluate',side_effect=evaluate),patch('train._save_curves'),patch('builtins.print'):
            train_val(args,H,loader,val,object(),None,12,S,nfd=nfd,fcd=fcd)
        self.assertEqual(nets[0].calls,12);self.assertEqual(nets[0].bn.num_batches_tracked.item(),12)
        self.assertFalse(json.loads((path/'metrics.json').read_text())['query_evaluated'])
        self.assertFalse(any('teacher_logits' in k or 'head.' in k for k in nets[0].state_dict()))
        return json.loads((path/'training_history.json').read_text()),nets[0].state_dict()

    def test_zero_exact_nfd_and_enabled_single_forward_all_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base,bs=self.run_tiny(root/'base')
            off,os=self.run_tiny(root/'off','neighbor',0)
            self.assertEqual([r['loss'] for r in base],[r['loss'] for r in off])
            for key in bs:self.assertTrue(torch.equal(bs[key],os[key]),key)
            for mode in ('ce','plain','neighbor'):
                history,_=self.run_tiny(root/mode,mode)
                self.assertEqual(history[0]['loss'],base[0]['loss'])
                self.assertGreater(history[1]['loss_fcd'],0)
                self.assertIn('fcd_sampled_rank_clean_ratio',history[1])
                self.assertTrue((root/mode/'fcd_selected_auxiliary_head.pt').exists())


class EntryTests(unittest.TestCase):
    def test_actual_parent_and_fcd_main_dispatch(self):
        import run
        with tempfile.TemporaryDirectory() as folder:
            args,argv=nfd_tests.EntryTests().fixture(folder)
            argv[argv.index('b2nfd')]='b3fcd'
            cache=Path(folder)/'foundation.pt'
            argv+=['--fcd-cache',str(cache),'--fcd-topk','2']
            with patch('sys.argv',argv):child=run.load_config()
            child.split_hash=args.split_hash;child.output_dir=args.output_dir
            data=dataset_fixture();child.classifier_sha256=args.classifier_sha256
            cache_fixture(cache,child,data)
            loader=DataLoader(data,batch_size=3,generator=torch.Generator().manual_seed(60))
            with patch('run.load_config',return_value=child),patch('network.ClassifyNet',CacheTeacher), \
                 patch('run.load_data',return_value=(loader,None,None,None,None,0,0,0,12)), \
                 patch('run.GenerateSimilarityMatrix') as relation,patch('run.GenerateSemanticHashCenters') as centers, \
                 patch('run.train_val') as train,patch('builtins.print'):
                run.main()
            relation.assert_not_called();centers.assert_not_called()
            self.assertIsNotNone(train.call_args.kwargs['fcd'])
            self.assertIsNone(train.call_args.kwargs['npa'])
            manifest=json.loads((Path(child.run_dir)/'config.json').read_text())
            self.assertIn('fcd_cache_sha256',manifest);self.assertIn('nfd_basis_sha256',manifest)
            for extra in [['--fcd-temperature','nan'],['--fcd-gamma','2'],['--fcd-topk','3'],['--fcd-ramp-end','301']]:
                with patch('sys.argv',argv+extra),patch('sys.stderr'),self.assertRaises(SystemExit):run.load_config()


class PreparationTests(unittest.TestCase):
    def test_frozen_extraction_classifier_probe_and_gate(self):
        from scripts.prepare_foundation_teacher import extract,fit_classifier,fit_hash_probe,retrieval_metrics,teacher_check
        data=dataset_fixture();loader=DataLoader(data,batch_size=3)
        class Old(nn.Module):
            def forward_with_features(self,x):return x[:,:3],x
        foundation=nn.Linear(4,6).eval().requires_grad_(False)
        before=copy.deepcopy(foundation.state_dict())
        ids,labels,new,old,old_logits=extract(loader,foundation,Old(),torch.device('cpu'))
        self.assertEqual(ids.tolist(),list(range(12)))
        for k in before:torch.testing.assert_close(before[k],foundation.state_dict()[k],atol=0,rtol=0)
        head,logits,vlogits,acc=fit_classifier(new,labels,new,labels,3,torch.device('cpu'),2)
        self.assertEqual(logits.shape,(12,3));self.assertTrue(math_isfinite(acc))
        codes,vcodes=fit_hash_probe(new,labels,new,H,torch.device('cpu'),2)
        proxy=retrieval_metrics(codes,vcodes,labels,labels,3)
        self.assertEqual(set(proxy),{'mAP@ALL','mAP@100','mAP@1000'})
        report=teacher_check(F.one_hot(labels,3).float(),-F.one_hot(labels,3).float(),labels,
                              {k:.8 for k in proxy},{k:.5 for k in proxy})
        self.assertTrue(report['passed'])
        failed=teacher_check(old_logits,old_logits,labels,proxy,proxy)
        self.assertFalse(failed['passed'])

    def test_preparation_cli_exports_cpu_cache_without_query_or_download(self):
        from scripts import prepare_foundation_teacher as prepare
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture=root/'reference';fixture.mkdir()
            args,_=nfd_tests.EntryTests().fixture(fixture)
            reference=json.loads(Path(args.parent_b1_config).read_text())
            reference.update(variant='b2nfd',code_length=32)
            reference['num_workers']=0
            config=fixture/'nfd.json';config.write_text(json.dumps(reference))
            repo=root/'official';repo.mkdir();(repo/'hubconf.py').write_text('# synthetic local backbone')
            weights=root/'weights.pt';foundation=nn.Linear(4,6);torch.save(foundation.state_dict(),weights)
            train_data=dataset_fixture();val_data=dataset_fixture()
            val_data.tensors=(*val_data.tensors[:2],val_data.tensors[2]+12)
            val_data.indices=np.arange(12,24)
            train=DataLoader(train_data,batch_size=3);val=DataLoader(val_data,batch_size=3)
            class Old(nn.Module):
                def forward_with_features(self,x):return x[:,:3],x
            output=root/'prepared'
            def load_data(options):
                Path(options.output_dir).mkdir()
                return train,train,val,object(),object(),0,0,0,12
            argv=['prepare','--nfd-config',str(config),'--foundation-repo',str(repo),
                  '--foundation-weights',str(weights),'--output-dir',str(output),'--probe-steps','2','--cpu']
            # The information-gate formula has its own independent test above.
            with patch('sys.argv',argv),patch('torch.hub.load',return_value=foundation) as hub, \
                 patch('data.data_loader.load_data',side_effect=load_data), \
                 patch('utils.frozen_b1.load_frozen_b1',return_value=(S,H.repeat(1,8))), \
                 patch('utils.neighbor_feature_distill.load_nfd',return_value=argparse.Namespace(teacher=Old())), \
                 patch.object(prepare,'teacher_check',return_value={'passed':True}),patch('builtins.print'):
                prepare.main()
            self.assertEqual(hub.call_args.kwargs,dict(source='local',pretrained=False))
            payload=torch.load(output/'foundation_cache.pt',weights_only=False)
            self.assertEqual(payload['ids'].tolist(),list(range(12)))
            self.assertEqual(payload['logits'].device.type,'cpu')
            self.assertFalse(json.loads((output/'teacher_report.json').read_text())['query_evaluated'])


class LauncherTests(unittest.TestCase):
    def test_explicit_references_and_three_run_summary(self):
        from scripts import run_i3_fcd as launch
        from tests import test_i3_launcher as npa_tests
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            for bit in (16,32,64):
                path,_=npa_tests.LauncherTests().fixture(root/str(bit),bit);paths.append(path)
            cache=root/'cache.pt';torch.save({'teacher_check':{'passed':True}},cache)
            output=root/'output';argv=['launch','--teacher-cache',str(cache),'--output-root',str(output),'--evaluate-query']
            for path in paths:argv+=['--nfd-config',str(path)]
            with patch('sys.argv',argv+['--dry-run']),patch('subprocess.run') as process,patch('builtins.print'):
                launch.main()
            process.assert_not_called();self.assertFalse(output.exists())
            def fake_run(command,**unused):
                self.assertEqual(command[command.index('--variant')+1],'b3fcd')
                self.assertIn('--fcd-cache',command);self.assertIn('--teacher-cache',command)
                self.assertNotIn('--npa-mode',command)
                run=Path(command[command.index('--output-dir')+1])/'runs'/'synthetic';run.mkdir(parents=True)
                (run/'metrics.json').write_text(json.dumps({'selected_epoch':10,'query_evaluated':True,
                    'best_validation_mAP@ALL':.5,'query_metrics':{'mAP@ALL':.6,'mAP@100':.7,'mAP@1000':.65}}))
            with patch('sys.argv',argv),patch('subprocess.run',side_effect=fake_run) as process,patch('builtins.print'):
                launch.main()
            self.assertEqual(process.call_count,3)
            import csv
            with (output/'summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual([r['bits'] for r in rows],['16','32','64'])
            torch.save({'teacher_check':{'passed':False}},cache)
            with patch('sys.argv',argv+['--output-root',str(root/'failed')]),patch('sys.stderr'),self.assertRaises(SystemExit):
                launch.main()


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__=='__main__':unittest.main()
