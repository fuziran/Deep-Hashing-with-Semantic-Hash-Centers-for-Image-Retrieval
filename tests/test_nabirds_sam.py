"""SAM migration, historical NABirds protocol and independent-query checks."""
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
from torch.utils.data import DataLoader, TensorDataset

from tests import test_nabirds_nfd as toy
from scripts import run_nabirds_b_sam as launch
from scripts import evaluate_nabirds_sam_query as evaluator
from utils.neighbor_feature_distill import NeighborhoodFeatureDistillation
from utils.sam_nfd import sam_nfd_step, VERSION
from utils.experiment import file_sha256, selection_receipt, source_sha256


def tiny_args(path, bits=4, rho=.05):
    return argparse.Namespace(variant="b2nfd", device=torch.device("cpu"), code_length=bits,
        num_classes=3, lr=1e-4, epoch=3, test_map=1, lambd=1e-4, nfd_weight=.01,
        nfd_warmup=1, nfd_ramp_end=2, skip_query=True, info="synthetic",
        run_dir=str(path), sam_rho=rho, sam_version=VERSION)


class SAMTests(unittest.TestCase):
    def fixture(self):
        from train import CSQLoss
        torch.manual_seed(60)
        args = tiny_args(".")
        net = toy.TinyNet(4)
        nfd = NeighborhoodFeatureDistillation(toy.Teacher(), toy.S, toy.P, topk=2)
        loss = CSQLoss(args, 4, toy.H)
        images, ids = torch.randn(6, 4), torch.arange(6)
        labels = F.one_hot(ids % 3, 3).float()
        return args, net, nfd, loss, images, labels, ids

    def test_manual_sam_gradient_and_bn_once(self):
        args, net, nfd, loss, x, y, ids = self.fixture()
        manual = copy.deepcopy(net)
        loss(manual(x), y, ids, args).backward()
        norm = torch.stack([p.grad.norm() for p in manual.parameters()]).norm()
        original = [p.detach().clone() for p in manual.parameters()]
        with torch.no_grad():
            for p in manual.parameters():
                p.add_(p.grad * args.sam_rho / (norm + 1e-12))
        manual.zero_grad(set_to_none=True)
        loss(manual(x), y, ids, args).backward()
        expected = [w - .01 * p.grad for w, p in zip(original, manual.parameters())]
        sam_nfd_step(net, torch.optim.SGD(net.parameters(), lr=.01), loss, nfd, x, y, ids, args, 0.)
        for p, value in zip(net.parameters(), expected):
            torch.testing.assert_close(p, value, rtol=1e-6, atol=1e-7)
        self.assertEqual(net.bn.num_batches_tracked.item(), 1)
        self.assertEqual(nfd.teacher.calls, 0)
        self.assertAlmostEqual(loss.components["sam_perturbation_norm"], .05, places=6)

    def test_dropout_rng_bn_teacher_reuse_and_rollback(self):
        args, _, nfd, loss, x, y, ids = self.fixture()
        class Stochastic(toy.TinyNet):
            def __init__(self):
                super().__init__(4)
                self.dropout, self.masks = nn.Dropout(.25), []
            def forward_with_features(self, images):
                mask = self.dropout(torch.ones_like(images))
                self.masks.append(mask.detach().clone())
                return super().forward_with_features(images * mask)
        net = Stochastic()
        ordinary = copy.deepcopy(net)
        torch.manual_seed(99)
        ordinary(x)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(99)
        sam_nfd_step(net, torch.optim.RMSprop(net.parameters(), lr=1e-4), loss, nfd, x, y, ids, args, .01, True)
        self.assertTrue(torch.equal(net.masks[0], net.masks[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in net.bn.named_buffers():
            torch.testing.assert_close(value, dict(ordinary.bn.named_buffers())[key], rtol=0, atol=0)
        self.assertEqual(nfd.teacher.calls, 1)
        self.assertFalse(nfd.teacher.training)
        self.assertTrue(all(p.grad is None for p in nfd.teacher.parameters()))
        args, net, nfd, loss, x, y, ids = self.fixture()
        original = [p.detach().clone() for p in net.parameters()]
        forward, calls = loss.forward, []
        def fail(*values):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError("synthetic second pass")
            return forward(*values)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4)
        with patch.object(loss, "forward", side_effect=fail), patch.object(optimizer, "step") as update, \
                self.assertRaisesRegex(FloatingPointError, "second pass"):
            sam_nfd_step(net, optimizer, loss, nfd, x, y, ids, args, .01)
        update.assert_not_called()
        for p, value in zip(net.parameters(), original):
            self.assertTrue(torch.equal(p, value))
        self.assertEqual(net.bn.num_batches_tracked.item(), 1)

    def test_zero_radius_legacy_reduction_and_three_lengths_without_query(self):
        from train import train_val
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def train(name, bits, rho, legacy=False):
                destination = root / name
                destination.mkdir()
                args = tiny_args(destination, bits, rho)
                if legacy:
                    del args.sam_rho, args.sam_version
                torch.manual_seed(60)
                nfd = NeighborhoodFeatureDistillation(toy.Teacher(), toy.S, toy.P, topk=2)
                nets = []
                def factory(code_length):
                    net = toy.TinyNet(code_length)
                    nets.append(net)
                    return net
                args.net = factory
                loader = DataLoader(TensorDataset(torch.linspace(-1, 1, 48).reshape(12, 4),
                    F.one_hot(torch.arange(12) % 3, 3).float(), torch.arange(12)), batch_size=3,
                    shuffle=True, generator=torch.Generator().manual_seed(60))
                val, query = object(), object()
                def evaluate(selected, *unused):
                    self.assertIs(selected, val)
                    return ({k: .5 for k in launch.METRICS}, None, None, None, None, None)
                with patch("train._evaluate", side_effect=evaluate), patch("train._save_curves"), patch("builtins.print"):
                    train_val(args, toy.H.repeat(1, bits // 4), loader, val, query, None, 12, nfd=nfd)
                self.assertIs(json.loads((destination / "metrics.json").read_text())["query_evaluated"], False)
                return nets[0], nfd, json.loads((destination / "training_history.json").read_text())
            old, _, old_history = train("legacy", 16, 0., True)
            off, _, off_history = train("zero", 16, 0.)
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, off.state_dict()[key]), key)
            self.assertEqual([r["loss"] for r in old_history], [r["loss"] for r in off_history])
            for bit in (16, 32, 64):
                net, nfd, history = train(str(bit), bit, .05)
                self.assertEqual(net.hash_layer.weight.shape[0], bit)
                self.assertEqual(net.bn.num_batches_tracked.item(), 12)
                self.assertEqual(nfd.teacher.calls, 8)
                self.assertGreater(history[-1]["sam_perturbation_norm"], 0)
                self.assertIn("sampled_nfd_clean_ratio", history[-1])

    def test_cli_requires_validation_only_sam_and_offline_model_loading(self):
        from run import load_config
        base = ["run.py", "--dataset", "NABirds-B", "--variant", "b2nfd", "--cpu", "--num-classes", "555",
                "--parent-b1-config", "parent", "--similarity-cache", "relation", "--center-cache", "centers", "--teacher-cache", "teacher"]
        for extra in (("--sam-rho", ".05"), ("--sam-rho", "nan"), ("--sam-rho", "-1")):
            with patch("sys.argv", base + list(extra)), patch("sys.stderr"), self.assertRaises(SystemExit):
                load_config()
        with patch("sys.argv", base + ["--sam-rho", ".05", "--skip-query"]):
            self.assertEqual(load_config().sam_rho, .05)
        import network
        with patch("network._build_resnet", side_effect=AssertionError("No pretrained weights")):
            model = network.ResNet(16, pretrained=False)
        self.assertEqual(model.hash_layer.weight.shape, (16, 512))


class LoaderIdentityTests(unittest.TestCase):
    def test_file_change_changes_content_identity_without_resplitting(self):
        from PIL import Image
        from data.data_loader import load_data
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, count in (("train", 4), ("query", 1), ("database", 1)):
                for name in ("bird_a", "bird_b", "bird_c"):
                    directory = root / "images" / split / name
                    directory.mkdir(parents=True)
                    for i in range(count):
                        Image.new("RGB", (8, 8)).save(directory / f"{i}.jpg")
            args = argparse.Namespace(dataset="NABirds-B", root=str(root), num_classes=3, seed=60,
                val_per_class=2, output_dir=str(root / "out"), device=torch.device("cpu"),
                resize_size=8, crop_size=8, batch_size=2, num_workers=0)
            load_data(args)
            old_split, old_content = args.split_hash, args.dataset_content_sha256
            Image.new("RGB", (8, 8), "red").save(root / "images" / "query" / "bird_a" / "0.jpg")
            load_data(args)
            self.assertEqual(old_split, args.split_hash)
            self.assertNotEqual(old_content, args.dataset_content_sha256)


class LauncherTests(unittest.TestCase):
    def fixture(self, root):
        configs, paths = {}, []
        for bit in (16, 32, 64):
            directory = root / f"{bit}bit"
            directory.mkdir(parents=True)
            config = {key: 1 for key in launch.SHARED}
            config.update(variant="b2nfd", dataset="NABirds-B", root=str(root / "images"), seed=60, num_classes=555, code_length=bit,
                lr=1e-4, epoch=300, classify_epoch=300, nfd_alpha=.75, nfd_weight=.01, nfd_topk=5,
                nfd_warmup=20, nfd_ramp_end=40, val_per_class=2, paired_stage_seeds=False,
                non_deterministic=False, topK=[-1, 100, 1000], split_hash="structural",
                nfd_teacher_sha256="teacher", nfd_basis_sha256="basis", center_content_sha256=f"center{bit}")
            for key in launch.ARTIFACTS:
                file = directory / (key + ".pt")
                file.write_bytes(b"synthetic cache")
                config[key] = str(file)
            path = directory / "config.json"
            path.write_text(json.dumps(config))
            history = [dict(epoch=e, validation={k: .4 + e / 10000 for k in launch.METRICS}) for e in range(1, 301)]
            (directory / "training_history.json").write_text(json.dumps(history))
            # The original NABirds metrics had no query_evaluated flag.
            (directory / "metrics.json").write_text(json.dumps(dict(selected_epoch=300, query_metrics={k: .5 for k in launch.METRICS})))
            configs[bit] = config
            paths.append(path)
        weights = root / "hub" / "checkpoints" / "resnet34-b627a593.pth"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"synthetic ImageNet weights")
        options = argparse.Namespace(output_root=str(root / "out"), baseline_root=str(root), nfd_config=[], bits=[16, 32, 64], gpu=0)
        return configs, paths, options

    def test_all_lengths_inherit_nabirds_parameters_and_only_sam(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs, paths, options = self.fixture(root)
            class Forbidden:
                def __iter__(self):
                    raise AssertionError("No loader may be iterated in preflight")
            def load(args):
                args.dataset_content_sha256 = "contents"
                args.dataset_counts = dict(train=10, validation=1110, query=6, database=9)
                return tuple(Forbidden() for _ in range(9))
            with patch("data.data_loader.load_data", side_effect=load) as data, \
                    patch("utils.frozen_b1.load_frozen_b1") as parent, \
                    patch("utils.neighbor_feature_distill.load_nfd"), \
                    patch("torch.hub.get_dir", return_value=str(root / "hub")):
                plan = launch.make_plan(options)
            self.assertEqual(data.call_count, 1)
            self.assertEqual(parent.call_count, 3)
            for item in plan["items"]:
                command, bit = item["command"], item["bits"]
                self.assertIn("--skip-query", command)
                self.assertNotIn("--paired-stage-seeds", command)  # Preserve actual NABirds legacy setting.
                self.assertEqual(command[command.index("--sam-rho") + 1], "0.05")
                self.assertEqual(command[command.index("--num-classes") + 1], "555")
                self.assertEqual(command[command.index("--lr") + 1], "0.0001")
                self.assertEqual(command[command.index("--root") + 1], configs[bit]["root"])
                self.assertEqual(command[command.index("--center-cache") + 1], configs[bit]["center_cache"])
                self.assertFalse(item["historical"]["query_evaluated_flag_present"])
            self.assertFalse(Path(options.output_root).exists())
            (paths[0].parent / "copy").mkdir()
            (paths[0].parent / "copy" / "config.json").write_text(paths[0].read_text())
            with self.assertRaisesRegex(ValueError, "Multiple"):
                launch.references(root, [], [16, 32, 64])

    def test_missing_reference_and_dry_run_never_start_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, paths, options = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Missing"):
                launch.references(root, [str(paths[0])], [16, 32])
            plan = dict(items=[dict(bits=32, reference_config=str(paths[1]), command=["python", "run.py"])],
                        counts={}, split_hash="structural", dataset_content_sha256="content", note="synthetic")
            with patch("sys.argv", ["launcher", "--output-root", options.output_root, "--dry-run"]), \
                    patch.object(launch, "make_plan", return_value=plan), patch.object(launch, "run_process") as run, patch("builtins.print"):
                launch.main()
            run.assert_not_called()
            self.assertFalse(Path(options.output_root).exists())


class TinyStudent(nn.Module):
    def __init__(self, bits=4, pretrained=False):
        super().__init__()
        if pretrained:
            raise AssertionError("Query requested pretrained weights")
        self.hash_layer = nn.Linear(4, bits, bias=False)
        with torch.no_grad():
            self.hash_layer.weight.copy_(torch.eye(bits, 4))
    def forward(self, images):
        return self.hash_layer(images)


class QueryCoordinatorTests(unittest.TestCase):
    def test_all_three_sam_queries_preflight_before_any_query_and_no_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root, events = Path(folder), []
            plan = dict(dataset_content_sha256="content")
            trained = dict(results=[dict(bits=bit, run_dir=str(root / f"{bit}bit" / "runs" / "sam")) for bit in (16, 32, 64)])
            def preflight(options):
                self.assertTrue(options.dry_run)
                events.append(("check", options.run_dir))
                if len(events) == 3:
                    raise ValueError("last model failed preflight")
                return dict(dataset_content_sha256="content")
            with patch.object(launch, "check_plan", return_value=plan), patch.object(launch, "training_report", return_value=trained), \
                    patch.object(evaluator, "evaluate", side_effect=preflight), patch.object(launch, "run_process") as run, \
                    self.assertRaisesRegex(ValueError, "last model"):
                launch.evaluate_queries(root, 0)
            run.assert_not_called()
            self.assertFalse((root / "query").exists())
            events.clear()
            def successful(options):
                self.assertTrue(options.dry_run)
                events.append("check")
                return dict(dataset_content_sha256="content")
            def run(command, log):
                self.assertEqual(events[:3], ["check"] * 3)
                self.assertNotIn("control", str(log))
                self.assertIn("evaluate_nabirds_sam_query.py", command[1])
                events.append(log.stem)
            with patch.object(launch, "check_plan", return_value=plan), patch.object(launch, "training_report", return_value=trained), \
                    patch.object(evaluator, "evaluate", side_effect=successful), patch.object(launch, "run_process", side_effect=run), \
                    patch.object(launch, "query_report", return_value={}), patch("builtins.print"):
                launch.evaluate_queries(root, 0)
            self.assertEqual(events, ["check"] * 3 + ["16bit", "32bit", "64bit"])
            jobs = launch.read_json(root / "query" / "query_plan.json")
            self.assertEqual([job["bits"] for job in jobs], [16, 32, 64])

    def test_query_summary_uses_dynamic_counts_historical_query_and_registered_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = dict(bits=[16, 32, 64], source_sha256="source", split_hash="split",
                dataset_content_sha256="contents", counts=dict(query=7, database=19), items=[])
            jobs, trained = [], dict(results=[])
            for bit in plan["bits"]:
                run, output = root / f"run{bit}", root / "query" / f"{bit}bit"
                output.mkdir(parents=True)
                (output / "retrieval_outputs.pt").write_bytes(b"synthetic retrieval")
                plan["items"].append(dict(bits=bit, historical=dict(query_metrics={k: .5 for k in launch.METRICS})))
                jobs.append(dict(bits=bit, run_dir=str(run), destination=str(output)))
                trained["results"].append(dict(bits=bit, run_dir=str(run)))
                payload = dict(query_evaluated=True, selected_epoch=300, counts=plan["counts"],
                    dataset_content_sha256="contents", split_hash="split", evaluation_source_sha256="source",
                    retrieval_outputs_sha256=file_sha256(output / "retrieval_outputs.pt"),
                    query_metrics={k: .52 for k in launch.METRICS})
                (output / "metrics.json").write_text(json.dumps(payload))
            query_plan = root / "query" / "query_plan.json"
            query_plan.write_text(json.dumps(jobs))
            def validate(run):
                return dict(code_length=int(run.name[3:]), sam_rho=.05), dict(selected_epoch=300), {}, {}
            with patch.object(launch, "check_plan", return_value=plan), patch.object(launch, "training_report", return_value=trained), \
                    patch.object(evaluator, "validate_run", side_effect=validate):
                report = launch.query_report(root)
                self.assertEqual(report["counts"], dict(query=7, database=19))
                for delta in report["sam_minus_historical_nfd"].values():
                    self.assertAlmostEqual(delta["mAP@ALL"], .02)
                jobs[0]["run_dir"] = str(root / "unregistered")
                query_plan.write_text(json.dumps(jobs))
                with self.assertRaisesRegex(ValueError, "registered"):
                    launch.query_report(root)


class QueryTests(unittest.TestCase):
    def fixture(self, root):
        run, raw = root / "run", root / "raw"
        run.mkdir()
        for name in ("train", "query", "database"):
            (raw / name).mkdir(parents=True)
        config = dict(variant="b2nfd", dataset="NABirds-B", num_classes=555, code_length=4, seed=60,
            epoch=3, topK=[-1, 100, 1000], source_sha256=source_sha256(), split_hash="structural",
            dataset_content_sha256="content", non_deterministic=False, root=str(raw), run_id="synthetic",
            batch_size=2, num_workers=0, sam_rho=.05, paired_stage_seeds=False)
        (run / "config.json").write_text(json.dumps(config))
        history = [dict(epoch=e, validation={k: value for k in launch.METRICS}) for e, value in ((1, .4), (2, .6), (3, .6))]
        (run / "training_history.json").write_text(json.dumps(history))
        torch.save(TinyStudent().state_dict(), run / "best_model_state.pt")
        (run / "metrics.json").write_text(json.dumps(dict(selected_epoch=2, selection_metric="validation mAP@ALL",
            **{"best_validation_mAP@ALL": .6}, query_evaluated=False, **selection_receipt(run))))
        def dataset(images, classes, offset):
            ids = torch.arange(offset, offset + len(images))
            data = TensorDataset(torch.tensor(images).float(), F.one_hot(torch.tensor(classes), 555).float(), ids)
            data.indices = ids.numpy()
            data.labels = np.zeros(offset + len(images), dtype=np.int64)
            data.labels[data.indices] = classes
            return DataLoader(data, batch_size=2)
        query = dataset([[1, 1, -1, -1], [0, 1, 1, -1]], [0, 1], 0)
        gallery = dataset([[1, 1, -1, -1], [-1, 1, 1, -1], [1, 1, -1, -1], [-1, -1, 1, 1]], [0, 1, 2, 1], 2)
        def load(args):
            args.split_hash, args.dataset_content_sha256 = "structural", "content"
            class Forbidden:
                def __iter__(self):
                    raise AssertionError("Query evaluator iterated training or validation")
            return Forbidden(), Forbidden(), Forbidden(), query, gallery, 0, 0, 2, 4
        options = argparse.Namespace(run_dir=str(run), output_dir=None, data_root=None, cpu=True, gpu=0, dry_run=False)
        return run, options, load

    def test_query_metrics_ids_receipts_and_original_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            run, options, load = self.fixture(Path(folder))
            original = {name: file_sha256(run / name) for name in ("config.json", "metrics.json", "training_history.json", "best_model_state.pt")}
            with patch.object(evaluator, "ResNet", TinyStudent), patch.object(evaluator, "load_data", side_effect=load), \
                    patch("torch.optim.RMSprop", side_effect=AssertionError("No optimizer in query")), \
                    patch("torch.hub.load", side_effect=AssertionError("No teacher in query")):
                report = evaluator.evaluate(options)
            self.assertTrue(report["query_evaluated"])
            self.assertEqual(report["selected_epoch"], 2)
            self.assertEqual(report["counts"], dict(query=2, database=4))
            for name, value in original.items():
                self.assertEqual(file_sha256(run / name), value)
            output = torch.load(run / "final_query" / "retrieval_outputs.pt", weights_only=False)
            self.assertEqual(output["query_ids"].tolist(), [0, 1])
            self.assertEqual(output["database_ids"].tolist(), [2, 3, 4, 5])

    def test_dry_run_never_forwards_and_data_and_receipt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run, options, load = self.fixture(Path(folder))
            options.dry_run = True
            with patch.object(evaluator, "ResNet", TinyStudent), patch.object(evaluator, "load_data", side_effect=load), \
                    patch.object(TinyStudent, "forward", side_effect=AssertionError("No forward in dry run")):
                self.assertFalse(evaluator.evaluate(options)["query_evaluated"])
            self.assertFalse((run / "final_query").exists())
            def changed(args):
                result = load(args)
                args.dataset_content_sha256 = "changed-file"
                return result
            with patch.object(evaluator, "ResNet", TinyStudent), patch.object(evaluator, "load_data", side_effect=changed), \
                    self.assertRaisesRegex(ValueError, "content/labels"):
                evaluator.evaluate(options)
            with (run / "config.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "receipt"):
                evaluator.validate_run(run)


if __name__ == "__main__":
    unittest.main()
