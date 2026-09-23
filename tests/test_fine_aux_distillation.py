"""Synthetic fine-label auxiliary and distillation integration tests."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tests.test_b2nfd_feature_distill import TinyNet, Teacher, S, P, H
from tests import test_b2nfd_feature_distill as nfd_tests
from utils.fine_aux import build_fine_aux, fine_aux_loss, fine_aux_weight
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation

class FineAuxTests(unittest.TestCase):
    def test_rng_gradients_labels_and_schedule(self):
        torch.manual_seed(60)
        state = torch.get_rng_state().clone()
        head = build_fine_aux(4, 3, 60, "cpu")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        net = TinyNet(4)
        _, features = net.forward_with_features(torch.randn(6, 4))
        labels = F.one_hot(torch.arange(6) % 3, 3).float()
        loss, _ = fine_aux_loss(head, features, labels)
        loss.backward()
        self.assertGreater(net.hidden.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(net.hash_layer.weight.grad)
        with self.assertRaises(ValueError): fine_aux_loss(head, features, torch.ones(6, 3))
        self.assertEqual([fine_aux_weight(e, .1, 1, 3) for e in [1,2,3]], [0,.05,.1])

    def run_tiny(self, path, alpha, weight=None):
        from train import train_val
        path.mkdir()
        torch.manual_seed(60)
        teacher = NeighborhoodFeatureDistillation(Teacher(), S, P, alpha=alpha, topk=2)
        frozen = {k: v.clone() for k,v in teacher.teacher.state_dict().items()}
        x = torch.linspace(-1,1,48).reshape(12,4)
        loader = DataLoader(TensorDataset(x, F.one_hot(torch.arange(12)%3,3).float(),
                                         torch.arange(12)), batch_size=3, shuffle=True,
                            generator=torch.Generator().manual_seed(60))
        net = TinyNet(4)
        initial = {k:v.clone() for k,v in net.state_dict().items()}
        args = argparse.Namespace(variant="b2nfd", seed=60, device=torch.device("cpu"),
             code_length=4,num_classes=3,lr=7e-5,epoch=3,test_map=1,lambd=1e-4,
             nfd_weight=.01,nfd_warmup=1,nfd_ramp_end=2,skip_query=True,
             info="synthetic",run_dir=str(path),net=lambda bits:net,
             fine_aux_warmup=0,fine_aux_ramp_end=1)
        if weight is not None: args.fine_aux_weight=weight
        validation=object()
        def evaluate(loader,*unused):
            self.assertIs(loader,validation)
            return ({"mAP@ALL":.5,"mAP@100":.6,"mAP@1000":.55},None,None,None,None,None)
        with patch("train._evaluate",side_effect=evaluate),patch("train._save_curves"),patch("builtins.print"):
            train_val(args,H,loader,validation,object(),None,12,S,nfd=teacher)
        for k,v in frozen.items(): self.assertTrue(torch.equal(v,teacher.teacher.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in teacher.teacher.parameters()))
        self.assertEqual(net.calls,12)
        self.assertEqual(net.bn.num_batches_tracked.item(),12)
        self.assertFalse(torch.equal(initial["hidden.weight"],net.hidden.weight))
        saved=torch.load(path/"best_model_state.pt",weights_only=True)
        TinyNet(4).load_state_dict(saved,strict=True)
        self.assertFalse(json.loads((path/"metrics.json").read_text())["query_evaluated"])
        return json.loads((path/"training_history.json").read_text()),net.state_dict()

    def test_both_modes_disabled_equivalence_and_saved_head(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for alpha in (0,.75):
                old, old_state = self.run_tiny(root/f"old{alpha}",alpha)
                off, off_state = self.run_tiny(root/f"off{alpha}",alpha,0)
                on, on_state = self.run_tiny(root/f"on{alpha}",alpha,.1)
                self.assertEqual([r["loss"] for r in old],[r["loss"] for r in off])
                for key in old_state: self.assertTrue(torch.equal(old_state[key],off_state[key]),key)
                self.assertFalse(torch.equal(on_state["hidden.weight"],off_state["hidden.weight"]))
                self.assertGreater(on[0]["loss_fine_aux"],0)
                self.assertGreater(on[1]["loss_nfd"],0)
                self.assertAlmostEqual(on[1]["loss"],on[1]["loss_center"]+1e-4*on[1]["loss_quant"]
                                       +.1*on[1]["loss_fine_aux"]+.01*on[1]["loss_nfd"],places=5)
                self.assertFalse((root/f"off{alpha}"/"best_fine_aux_state.pt").exists())
                payload=torch.load(root/f"on{alpha}"/"best_fine_aux_state.pt",weights_only=True)
                self.assertEqual(payload["epoch"],1)
                initial=build_fine_aux(4,3,60,"cpu").state_dict()
                self.assertFalse(torch.equal(initial["weight"],payload["state_dict"]["weight"]))

    def test_main_identity_and_frozen_dispatch(self):
        import run
        with tempfile.TemporaryDirectory() as folder:
            args, _ = nfd_tests.EntryTests().fixture(folder)
            loader = argparse.Namespace(generator=torch.Generator())
            ids = []
            for weight in (0, .1, .2):
                args.fine_aux_weight = weight
                args.output_dir = str(Path(folder)/str(weight))
                with patch("run.load_config", return_value=args), \
                     patch("network.ClassifyNet", nfd_tests.CacheTeacher), \
                     patch("run.load_data", return_value=(loader,None,None,None,None,0,0,0,1)), \
                     patch("run.GenerateSimilarityMatrix") as relation, \
                     patch("run.GenerateSemanticHashCenters") as centers, \
                     patch("run.train_val") as train, patch("builtins.print"):
                    run.main()
                relation.assert_not_called()
                centers.assert_not_called()
                train.assert_called_once()
                self.assertIsNotNone(train.call_args.kwargs["nfd"])
                config=json.loads((Path(args.run_dir)/"config.json").read_text())
                self.assertEqual(config["fine_aux_weight"],weight)
                ids.append(args.run_id)
            self.assertEqual(len(set(ids)),3)

    def test_cli_validation(self):
        from run import load_config
        with tempfile.TemporaryDirectory() as folder:
            _, argv=nfd_tests.EntryTests().fixture(folder)
            with patch("sys.argv",argv+["--fine-aux-weight",".1"]):
                self.assertEqual(load_config().fine_aux_weight,.1)
            for extra in (["--fine-aux-weight","nan"],["--fine-aux-weight","-1"],
                          ["--fine-aux-warmup","40"],["--fine-aux-weight",".1","--fine-aux-ramp-end","301"]):
                with patch("sys.argv",argv+extra),patch("sys.stderr"),self.assertRaises(SystemExit): load_config()

if __name__ == "__main__": unittest.main()
