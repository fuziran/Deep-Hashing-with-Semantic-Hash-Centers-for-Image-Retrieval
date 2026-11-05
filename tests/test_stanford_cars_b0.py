import argparse
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
from PIL import Image

from data.data_loader import load_data
from scripts import run_stanford_cars_b0 as launch
from utils.experiment import file_sha256, source_sha256, write_run_manifest


class CarsLoaderTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder) / "images"
        for split, offset, count in (("train",0,3), ("query",3,1), ("database",4,1)):
            for c in range(196):
                directory = root / split / str(c)
                directory.mkdir(parents=True)
                for j in range(count):
                    value = c*8 + offset+j
                    Image.new("RGB", (8,8), (value%256, value//256, 19)).save(directory/f"{j}.png")
        args = argparse.Namespace(root=str(root.parent), dataset="Stanford_Cars-B", num_classes=196,
            val_per_class=2, seed=60, output_dir=str(Path(folder)/"out"), resize_size=8, crop_size=8,
            batch_size=8, num_workers=0, device=torch.device("cpu"))
        return root, args

    def test_original_splits_holdout_labels_and_content_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root,args = self.fixture(folder)
            result = load_data(args)
            train,relation,val,query,db = result[:5]
            self.assertEqual(result[-4:], (196,392,196,196))
            np.testing.assert_array_equal(train.dataset.indices, relation.dataset.indices)
            for left,right in ((train,val),(train,query),(train,db),(val,query),(val,db),(query,db)):
                self.assertFalse(set(left.dataset.indices) & set(right.dataset.indices))
            self.assertEqual(query.dataset[0][1].shape, (196,))
            self.assertEqual(query.dataset[0][1].sum().item(), 1)
            original_split,original_content = args.split_hash,args.dataset_content_sha256
            load_data(args)
            self.assertEqual(args.split_hash,original_split)
            self.assertEqual(args.dataset_content_sha256,original_content)
            Image.new("RGB",(8,8),(255,255,255)).save(root/"train"/"0"/"0.png")
            load_data(args)
            self.assertEqual(args.split_hash,original_split)
            self.assertNotEqual(args.dataset_content_sha256,original_content)
            manifest=json.loads(next(p for p in (Path(args.output_dir)/"splits").glob("*.json") if not p.name.endswith("_content_duplicates.json")).read_text())
            self.assertEqual(manifest["dataset_content_sha256"],args.dataset_content_sha256)

    def test_wrong_classes_rejected_and_original_content_duplicates_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root,args=self.fixture(folder)
            args.num_classes=555
            with self.assertRaisesRegex(ValueError,"196"):
                load_data(args)
            args.num_classes=196
            (root/"query"/"0"/"0.png").write_bytes((root/"train"/"0"/"0.png").read_bytes())
            result=load_data(args)
            self.assertEqual(result[-4:],(196,392,196,196))
            self.assertEqual(len(args.exact_content_duplicates),1)
            self.assertEqual(args.exact_content_duplicates[0]["first_split"],"train")
            self.assertEqual(args.exact_content_duplicates[0]["second_split"],"query")


class CarsProtocolTests(unittest.TestCase):
    def test_preflight_no_loader_iteration_and_correct_three_b0_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"models_ckpt").mkdir()
            (root/"models_ckpt"/"resnet34-b627a593.pth").write_bytes(b"test ImageNet")
            options=argparse.Namespace(root=str(root/"data"), output_root=str(root/"out"), gpu=0)
            class Forbidden:
                def __iter__(self):
                    raise AssertionError("Preflight iterated an image loader")
            def load(args):
                args.source_counts=dict(launch.PAPER_COUNTS)
                args.dataset_counts=dict(train=5488,validation=392,query=1958,database=8347)
                args.split_hash,args.dataset_content_sha256="split","content"
                return tuple(Forbidden() for _ in range(9))
            with patch.object(launch,"REPO",root), patch("data.data_loader.load_data", side_effect=load), patch("builtins.print"):
                plan=launch.preflight(options)
            self.assertEqual([x["bits"] for x in plan["items"]],[16,32,64])
            for row in plan["items"]:
                command=row["command"]
                for key,value in (("--variant","b0"),("--num-classes","196"),("--seed","60"),
                        ("--lr","0.0001"),("--epoch","300"),("--classify-epoch","300"),("--val-per-class","2")):
                    self.assertEqual(command[command.index(key)+1],value)
                self.assertIn("--paired-stage-seeds",command)
                self.assertNotIn("--sam-rho",command)
                self.assertNotIn("--nfd-weight",command)
            self.assertFalse(Path(options.output_root).exists())
            def wrong(args):
                result=load(args)
                args.source_counts["train"]=8144
                return result
            with patch.object(launch,"REPO",root), patch("data.data_loader.load_data", side_effect=wrong), patch("builtins.print"), self.assertRaisesRegex(ValueError,"curated Cars-B"):
                launch.preflight(options)

    def test_dry_run_never_launches_training_and_cli_is_b0_only(self):
        plan=dict(source_counts={},counts={},split_hash="split",dataset_content_sha256="content")
        with patch.object(launch,"preflight",return_value=plan), patch("subprocess.run") as subprocess_run, patch("builtins.print"):
            launch.main(["--root","unused","--output-root","unused","--dry-run"])
        subprocess_run.assert_not_called()
        from run import load_config
        with patch("sys.argv",["run.py","--cpu","--dataset","Stanford_Cars-B","--num-classes","196"]):
            args=load_config()
            self.assertEqual(args.val_per_class,2)
            self.assertEqual(args.variant,"b0")
        for variant in ("b1","b2nfd","sam"):
            with patch("sys.argv",["run.py","--cpu","--variant",variant]), patch("sys.stderr"), self.assertRaises(SystemExit):
                load_config()

    def test_hash_stage_initialization_and_order_ignore_stage1_random_consumption(self):
        import run
        captured=[]
        with tempfile.TemporaryDirectory() as folder:
            for consumed in (11,77):
                args=argparse.Namespace(**launch.configuration(folder), device=torch.device("cpu"),
                    output_dir=str(Path(folder)/str(consumed)), code_length=32, topK=[-1,100,1000],
                    non_deterministic=False, paired_stage_seeds=True, expected_dataset_sha256=None,
                    similarity_cache=None, force_recompute=False, stage="all", git_commit="test",
                    source_sha256=source_sha256(), info="test", net=lambda bit:nn.Linear(4,bit))
                def load(args):
                    args.split_hash,args.dataset_content_sha256="split","content"
                    args.dataset_counts=dict(train=6,validation=3,query=2,database=4)
                    args.source_counts=dict(train=9,query=2,database=4)
                    loader=DataLoader(TensorDataset(torch.zeros(6,4),torch.zeros(6,196),torch.arange(6)),
                        batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(60))
                    return loader,object(),object(),object(),object(),6,3,2,4
                def similarity(args,*unused):
                    torch.rand(consumed)
                    args.similarity_hash="similarity"
                    return torch.eye(196)
                def train(args,centers,loader,*unused):
                    net=args.net(args.code_length)
                    captured.append((net.weight.detach().clone(),next(iter(loader))[2]))
                with patch("run.load_config",return_value=args), patch("run.load_data",side_effect=load), \
                        patch("run.GenerateSimilarityMatrix",side_effect=similarity), \
                        patch("run.GenerateSemanticHashCenters",return_value=torch.ones(196,32)), patch("run.train_val",side_effect=train):
                    run.main()
        torch.testing.assert_close(captured[0][0],captured[1][0],rtol=0,atol=0)
        torch.testing.assert_close(captured[0][1],captured[1][1],rtol=0,atol=0)

    def test_cpu_manifest_on_gpu_host_and_report_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=argparse.Namespace(run_dir=str(root),device=torch.device("cpu"))
            with patch("torch.cuda.is_available",return_value=True), patch("torch.cuda.get_device_name",side_effect=AssertionError("CPU manifest queried GPU name")):
                write_run_manifest(args)
            args.device,args.gpu=torch.device("cuda:2"),2
            with patch("torch.cuda.is_available",return_value=True), patch("torch.cuda.get_device_name",return_value="test GPU"):
                write_run_manifest(args)
            manifest=json.loads((root/"config.json").read_text())
            self.assertEqual(manifest["gpu"],2)
            self.assertEqual(manifest["gpu_name"],"test GPU")
            settings=launch.configuration(root/"data")
            plan=dict(version=launch.VERSION,settings=settings,source_sha256=source_sha256(),
                split_hash="split",dataset_content_sha256="content",counts=dict(train=5488,validation=392,query=1958,database=8347))
            (root/"experiment_plan.json").write_text(json.dumps(plan))
            for bit in (16,32,64):
                run=root/"runs"/f"Stanford_Cars-B_B0_{bit}bit_seed60_test"
                run.mkdir(parents=True)
                config={**settings,"code_length":bit,"source_sha256":plan["source_sha256"],
                    "split_hash":"split","dataset_content_sha256":"content","paired_stage_seeds":True}
                history=[dict(epoch=i,validation={k:.6 if i in (275,300) else .5 for k in launch.METRICS}) for i in range(1,301)]
                (run/"config.json").write_text(json.dumps(config))
                (run/"training_history.json").write_text(json.dumps(history))
                (run/"best_model_state.pt").write_bytes(b"test checkpoint")
                (run/"retrieval_outputs.pt").write_bytes(b"test retrieval")
                metrics=dict(selected_epoch=275,selection_metric="validation mAP@ALL",best_validation_mAP_ALL=.6,
                    query_evaluated=True,counts=dict(query=1958,database=8347),query_metrics={k:.7 for k in launch.METRICS})
                metrics["best_validation_mAP@ALL"]=metrics.pop("best_validation_mAP_ALL")
                for key,name in (("checkpoint_sha256","best_model_state.pt"),("training_config_sha256","config.json"),
                        ("training_history_sha256","training_history.json"),("retrieval_outputs_sha256","retrieval_outputs.pt")):
                    metrics[key]=file_sha256(run/name)
                (run/"metrics.json").write_text(json.dumps(metrics))
            result=launch.report(root)
            self.assertEqual([r["bits"] for r in result["results"]],[16,32,64])
            self.assertEqual(result["counts"]["query"],1958)
            (run/"best_model_state.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"artifact mismatch"):
                launch.report(root)


class TinyB0TrainingTests(unittest.TestCase):
    def test_three_lengths_select_validation_then_query_once(self):
        from train import train_val
        class Tiny(nn.Module):
            def __init__(self,bit):
                super().__init__()
                self.hash_layer=nn.Linear(4,bit)
            def forward(self,x):
                return self.hash_layer(x)
        for bit in (16,32,64):
            with self.subTest(bit=bit), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                args=argparse.Namespace(net=Tiny,code_length=bit,num_classes=3,device=torch.device("cpu"),
                    lr=1e-4,epoch=2,test_map=1,lambd=1e-4,topK=[-1,100,1000],info="test",run_dir=str(root))
                (root/"config.json").write_text('{}')
                train=DataLoader(TensorDataset(torch.randn(6,4),F.one_hot(torch.arange(6)%3,3).float(),torch.arange(6)),batch_size=3)
                val=DataLoader(TensorDataset(torch.zeros(3,4)),batch_size=3)
                query=DataLoader(TensorDataset(torch.zeros(2,4)),batch_size=2)
                db=DataLoader(TensorDataset(torch.zeros(4,4)),batch_size=4)
                snapshots=[]
                def evaluate(loader,database,net,*unused):
                    if loader is val:
                        snapshots.append({k:v.detach().clone() for k,v in net.state_dict().items()})
                        value=.8 if len(snapshots)==1 else .2
                    else:
                        self.assertIs(loader,query)
                        self.assertEqual(len(snapshots),2)
                        for k,v in net.state_dict().items():
                            torch.testing.assert_close(v,snapshots[0][k],rtol=0,atol=0)
                        value=.6
                    return ({k:value for k in launch.METRICS},None,torch.ones(len(loader.dataset),bit),
                        F.one_hot(torch.zeros(len(loader.dataset),dtype=torch.long),3).float(),torch.ones(4,bit),
                        F.one_hot(torch.zeros(4,dtype=torch.long),3).float())
                with patch("train._evaluate",side_effect=evaluate) as score, patch("train.CalcTopMapPerQuery",return_value={}), patch("train._save_curves"), patch("builtins.print"):
                    result=train_val(args,torch.where(torch.randn(3,bit)>0,1.,-1.),train,val,query,db,4)
                self.assertEqual(score.call_count,3)
                self.assertEqual(result["selected_epoch"],1)
                self.assertTrue(result["query_evaluated"])
                self.assertEqual(result["counts"],dict(query=2,database=4))
                self.assertEqual(len(json.loads((root/"training_history.json").read_text())),2)


if __name__ == "__main__":
    unittest.main()
