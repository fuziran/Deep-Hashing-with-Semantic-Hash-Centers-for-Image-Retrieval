"""NFD+Mixup pilot identity, matched control, and query isolation."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_i3_nfd_mix import build_command
from scripts.run_i3_npa import SHARED, ARTIFACTS
from utils.semantic_mixup import SemanticMixup


class NFDMixTests(unittest.TestCase):
    def test_variant_requires_nfd_and_query_skip(self):
        from run import load_config
        argv = ["run.py", "--variant", "b3nmix", "--cpu", "--paired-stage-seeds",
                "--skip-query", "--parent-b1-config", "parent.json",
                "--similarity-cache", "relation.pt", "--center-cache", "centers.pt",
                "--teacher-cache", "teacher.pt"]
        with patch("sys.argv", argv):
            args = load_config()
        self.assertEqual(args.mix_budget, 2.)
        self.assertEqual(args.mix_mode, "semantic")
        with patch("sys.argv", [value for value in argv if value != "--skip-query"]), \
             patch("sys.stderr"), self.assertRaises(SystemExit):
            load_config()

    def test_uniform_control_has_same_center_distance_cap(self):
        centers = torch.tensor([[1., 1., 1., 1.], [-1., -1., 1., 1.],
                                [1., -1., -1., 1.]])
        similarity = torch.tensor([[1., .7, .2], [.7, 1., .4], [.2, .4, 1.]])
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        for mode in ("semantic", "uniform_budgeted"):
            sampler = SemanticMixup(centers, similarity, 60, mode, budget=.25)
            anchors, partners, beta = sampler.sample(labels)
            distances = sampler.distance[labels[anchors], labels[partners]].float()
            self.assertEqual(len(anchors), len(labels))
            self.assertTrue((labels[anchors] != labels[partners]).all())
            self.assertTrue((beta * distances <= .25 + 1e-6).all())

    def test_launcher_inherits_nfd_and_skips_query(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            config = {key: 1 for key in SHARED}
            for key in ARTIFACTS:
                artifact = root / (key + ".pt")
                artifact.touch()
                config[key] = str(artifact)
            config.update(variant="b2nfd", dataset="cifar-100-new-seg", seed=60,
                          code_length=32, nfd_alpha=.75, nfd_weight=.01,
                          paired_stage_seeds=True, topK=[-1, 100, 1000],
                          root=str(raw), non_deterministic=False)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            options = argparse.Namespace(data_root=None, output_root=str(root / "out"),
                                         mode="uniform_budgeted", gpu=0, weight=.1,
                                         budget=2., warmup=20, ramp_end=40)
            command, _ = build_command(config_path, options)
            self.assertIn("--skip-query", command)
            self.assertIn("b3nmix", command)
            self.assertIn("uniform_budgeted", command)
            self.assertIn("--teacher-cache", command)

    def test_both_pilot_arms_accumulate_mix_and_nfd_gradients(self):
        from train import train_val

        class TinyNet(torch.nn.Module):
            def __init__(self, bits):
                super().__init__()
                self.feature = torch.nn.Linear(3, 3)
                self.hash_layer = torch.nn.Linear(3, bits)

            def forward_with_features(self, image):
                features = self.feature(image).tanh() + .5
                return self.hash_layer(features), features

            def forward(self, image):
                return self.forward_with_features(image)[0]

        class TinyNFD:
            def __init__(self):
                self.components = {}

            def __call__(self, features, images, targets):
                del images, targets
                loss = features.square().mean()
                self.components = {"loss_nfd": float(loss.detach())}
                return loss

        images = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                               [1., .2, 0.], [.2, 1., 0.]])
        labels = torch.nn.functional.one_hot(torch.tensor([0, 1, 0, 1]), 2).float()
        centers = torch.tensor([[1., 1.], [-1., -1.]])
        similarity = torch.tensor([[1., .5], [.5, 1.]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            for mode in ("semantic", "uniform_budgeted"):
                torch.manual_seed(60)
                loader = DataLoader(TensorDataset(images, labels, torch.arange(4)),
                                    batch_size=4, shuffle=False)
                run = Path(folder) / mode
                run.mkdir()
                args = argparse.Namespace(variant="b3nmix", device=torch.device("cpu"),
                                          code_length=2, num_classes=2, lr=7e-5,
                                          epoch=1, test_map=1, lambd=1e-4,
                                          nfd_weight=.01, nfd_warmup=0, nfd_ramp_end=1,
                                          mix_mode=mode, mix_weight=.1, mix_budget=.25,
                                          mix_seed=200063, mix_warmup=0, mix_ramp_end=1,
                                          skip_query=True, info="tiny", run_dir=str(run), net=TinyNet)
                with patch("train._evaluate", return_value=(
                        {"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55},
                        None, None, None, None, None)) as evaluate, \
                     patch("train._save_curves"), patch("builtins.print"):
                    train_val(args, centers, loader, object(), object(), None, 4,
                              similarity=similarity, nfd=TinyNFD())
                history = json.loads((run / "training_history.json").read_text())
                metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(evaluate.call_count, 1)
                self.assertFalse(metrics["query_evaluated"])
                self.assertGreater(history[0]["loss_nfd"], 0)
                self.assertGreater(history[0]["loss_mix"], 0)
                self.assertIn("mix_sampled_weighted_mix_grad_norm", history[0])


if __name__ == "__main__":
    unittest.main()
