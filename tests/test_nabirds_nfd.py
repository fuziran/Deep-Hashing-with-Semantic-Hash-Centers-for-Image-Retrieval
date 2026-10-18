"""Synthetic NABirds-B I2-NFD checks; no images or pretrained downloads."""

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

from utils.neighbor_feature_distill import (
    NeighborhoodFeatureDistillation, load_nfd, nfd_weight,
)

S = torch.tensor([[1., .8, .1], [.8, 1., .3], [.1, .3, 1.]])
P = torch.eye(3, 4)
H = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [-1., -1., -1., -1.]])


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
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

    def forward_with_features(self, x):
        features = self.hidden(self.bn(x)).tanh() + .4
        return self.hash_layer(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]


class CacheTeacher(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.feature = nn.Linear(4, 512)
        self.classify_layer = nn.Linear(512, classes)

    def forward_features(self, x):
        return self.feature(x)


class NFDTests(unittest.TestCase):
    def test_projection_formula_schedule_and_teacher_isolation(self):
        module = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2)
        self.assertEqual(module.neighbors.tolist(), [[1, 2], [0, 2], [1, 0]])
        self.assertEqual(module.ranks.tolist(), [2, 2, 2])
        student = torch.tensor([[1., 2., 3., 4.], [4., 2., 1., 3.]], requires_grad=True)
        teacher = torch.tensor([[2., 1., 4., 3.], [1., 2., 4., 3.]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = module.feature_loss(student, teacher, labels)
        expected = []
        for index, label in enumerate(labels):
            q = module.basis[label]
            a = (.25 * torch.eye(4) + .75 * q @ q.T) / (.25 + .75 * module.ranks[label] / 4)
            torch.testing.assert_close(torch.trace(a), torch.tensor(4.))
            error = F.normalize(student[index], dim=0) - F.normalize(teacher[index].detach(), dim=0)
            expected.append(error @ a @ error)
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward()
        self.assertGreater(student.grad.norm().item(), 0)
        self.assertIsNone(teacher.grad)
        module.train()
        self.assertFalse(module.teacher.training)
        self.assertTrue(all(not p.requires_grad for p in module.teacher.parameters()))
        for epoch, expected_weight in ((20, 0), (21, .0005), (40, .01)):
            self.assertAlmostEqual(nfd_weight(epoch), expected_weight)

    def test_plain_distillation_and_zero_rank_fallback(self):
        student, teacher = torch.randn(3, 4), torch.randn(3, 4)
        labels = torch.arange(3)
        ordinary = ((F.normalize(student, dim=1) - F.normalize(teacher, dim=1)) ** 2).sum(1).mean()
        self.assertEqual(NeighborhoodFeatureDistillation(Teacher(), S, P, alpha=0, topk=2)
                         .feature_loss(student, teacher, labels), ordinary)
        zero_rank = NeighborhoodFeatureDistillation(Teacher(), S, P[:1].repeat(3, 1), topk=2)
        self.assertEqual(zero_rank.ranks.tolist(), [0, 0, 0])
        self.assertEqual(zero_rank.feature_loss(student, teacher, labels), ordinary)

    def test_student_training_uses_one_forward_and_only_positive_weight_teacher_calls(self):
        from train import train_val
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = torch.linspace(-1, 1, 48).reshape(12, 4)
            loader = DataLoader(
                TensorDataset(x, F.one_hot(torch.arange(12) % 3, 3).float(), torch.arange(12)),
                batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(60),
            )
            validation = object()

            def evaluate(which, *unused):
                self.assertIs(which, validation)
                return ({"mAP@ALL": .5, "mAP@100": .6, "mAP@1000": .55}, None,
                        torch.zeros(1, 4), torch.zeros(1, 3),
                        torch.zeros(1, 4), torch.zeros(1, 3))

            def execute(name, variant, weight):
                torch.manual_seed(60)
                run_dir = root / name
                run_dir.mkdir()
                net = TinyNet(4)
                args = argparse.Namespace(
                    variant=variant, device=torch.device("cpu"), code_length=4,
                    num_classes=3, lr=7e-5, epoch=3, test_map=1, lambd=1e-4,
                    nfd_weight=weight, nfd_warmup=1, nfd_ramp_end=2,
                    info="synthetic", run_dir=str(run_dir), net=lambda bits: net,
                    topK=[-1, 100, 1000],
                )
                distiller = NeighborhoodFeatureDistillation(Teacher(), S, P, topk=2) \
                    if variant == "b2nfd" else None
                torch.manual_seed(60)
                loader.generator.manual_seed(60)
                with patch("train._evaluate", side_effect=evaluate), \
                     patch("train.CalcTopMapPerQuery", return_value=None), \
                     patch("train._save_curves"), patch("builtins.print"):
                    # Final query is redirected to the same synthetic validation object.
                    train_val(args, H, loader, validation, validation, None, 12, nfd=distiller)
                return json.loads((run_dir / "training_history.json").read_text()), net, distiller

            baseline, bnet, _ = execute("b1", "b1", 0)
            disabled, dnet, off = execute("off", "b2nfd", 0)
            self.assertEqual([x["loss"] for x in baseline], [x["loss"] for x in disabled])
            for key in bnet.state_dict():
                torch.testing.assert_close(bnet.state_dict()[key], dnet.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(off.teacher.calls, 0)
            active, _, on = execute("active", "b2nfd", .01)
            self.assertEqual(on.teacher.calls, 8)
            self.assertEqual(active[0]["loss"], baseline[0]["loss"])
            self.assertGreater(active[1]["loss_nfd"], 0)
            self.assertTrue((root / "active" / "best_model_state.pt").exists())

    def test_cli_limits_new_variant_to_nabirds_and_preserves_cifar(self):
        from run import load_config
        with patch("sys.argv", ["run.py", "--variant", "b2nfd", "--cpu"]), \
             patch("sys.stderr"), self.assertRaises(SystemExit):
            load_config()
        with patch("sys.argv", ["run.py", "--dataset", "cifar-100-new-seg", "--cpu"]):
            self.assertEqual(load_config().variant, "b0")

    def test_cpu_manifest_on_machine_with_available_cuda(self):
        from utils.experiment import write_run_manifest

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(run_dir=directory, device=torch.device("cpu"))
            with patch("utils.experiment.torch.cuda.is_available", return_value=True), \
                 patch("utils.experiment.torch.cuda.get_device_name",
                       side_effect=AssertionError("CPU manifest must not query a GPU")):
                write_run_manifest(args)
            manifest = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(manifest["device"], "cpu")
            self.assertTrue(manifest["cuda_available"])
            self.assertNotIn("gpu", manifest)

    def test_resnet_feature_path_keeps_forward_and_checkpoint_keys(self):
        import network
        from torchvision import models

        with patch.object(network, "_build_resnet", side_effect=lambda *a, **kw: models.resnet34(weights=None)):
            student = network.ResNet(4).eval()
        original_keys = set(student.state_dict())
        with torch.no_grad():
            images = torch.randn(2, 3, 32, 32)
            codes, features = student.forward_with_features(images)
            torch.testing.assert_close(student(images), codes, rtol=0, atol=0)
        self.assertEqual(features.shape, (2, 512))
        self.assertEqual(set(student.state_dict()), original_keys)

    def test_parent_cache_provenance_and_teacher_rng(self):
        from run import load_config
        from utils.experiment import build_cache_metadata, save_cache
        from utils.frozen_b1 import load_frozen_b1
        from utils.reliable_relation import relation_config, state_sha256, tensor_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["run.py", "--variant", "b2nfd", "--dataset", "NABirds-B", "--cpu",
                    "--root", str(root), "--num-classes", "3", "--code-length", "4",
                    "--nfd-topk", "2", "--parent-b1-config", str(root / "parent.json"),
                    "--similarity-cache", str(root / "relation.pt"),
                    "--center-cache", str(root / "centers.pt"),
                    "--teacher-cache", str(root / "teacher.pt")]
            with patch("sys.argv", argv):
                args = load_config()
            args.variant, args.split_hash, args.run_id = "b1", "split", "parent"
            state = CacheTeacher(3).state_dict()
            args.classifier_sha256 = state_sha256(state)
            args.similarity_content_sha256 = tensor_sha256(S)
            teacher_meta = build_cache_metadata(args, "classifier")
            relation_meta = build_cache_metadata(args, "similarity")
            args.similarity_hash = relation_meta["config_hash"]
            center_meta = build_cache_metadata(args, "centers")
            args.center_content_sha256 = tensor_sha256(H)
            save_cache(args.teacher_cache, state, teacher_meta)
            save_cache(args.similarity_cache, {
                "similarity": S, "similarity_sha256": tensor_sha256(S),
                "prototypes": torch.eye(3, 512), "classifier_sha256": args.classifier_sha256,
                "relation_config": relation_config(args),
            }, relation_meta)
            save_cache(args.center_cache, H, center_meta)
            parent = {key: value for key, value in vars(args).items() if key not in ("net", "device")}
            Path(args.parent_b1_config).write_text(json.dumps(parent), encoding="utf-8")
            args.variant, args.git_commit = "b2nfd", "new-child"
            _, centers = load_frozen_b1(args)
            torch.testing.assert_close(centers, H)
            before = torch.get_rng_state().clone()
            with patch("network.ClassifyNet", CacheTeacher):
                module = load_nfd(args)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(module.dim, 512)
            self.assertEqual(args.nfd_projection_ranks, [2, 2, 2])
            args.relation_kappa = 10
            with self.assertRaisesRegex(ValueError, "relation_kappa"):
                load_frozen_b1(args)
            args.relation_kappa = 30

            import run
            args.output_dir = str(root / "new_runs")
            loader = argparse.Namespace(generator=torch.Generator())
            with patch("run.load_config", return_value=args), \
                 patch("run.load_data", return_value=(loader, None, None, None, None,
                                                        0, 0, 0, 1)), \
                 patch("network.ClassifyNet", CacheTeacher), \
                 patch("run.GenerateSimilarityMatrix") as relation, \
                 patch("run.GenerateSemanticHashCenters") as centers_stage, \
                 patch("run.train_val") as training, patch("builtins.print"):
                run.main()
            relation.assert_not_called()
            centers_stage.assert_not_called()
            training.assert_called_once()
            self.assertIsNotNone(training.call_args.kwargs["nfd"])
            self.assertTrue((Path(args.run_dir) / "nfd_projection.pt").exists())
            self.assertEqual(json.loads((Path(args.run_dir) / "config.json").read_text())["variant"],
                             "b2nfd")

            payload = torch.load(args.teacher_cache, map_location="cpu")
            payload["value"]["feature.weight"][0, 0] += 1
            torch.save(payload, args.teacher_cache)
            with self.assertRaisesRegex(ValueError, "teacher content"):
                load_nfd(args)


if __name__ == "__main__":
    unittest.main()
