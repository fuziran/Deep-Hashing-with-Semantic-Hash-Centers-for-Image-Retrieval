"""Independent query evaluation: real metric path, synthetic data, no downloads."""
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

from scripts import evaluate_query as evaluator
from utils.experiment import file_sha256, selection_receipt, source_sha256
from utils.tools import CalcTopMapPerQuery, CalcTopMapWithPR


class TinyStudent(nn.Module):
    def __init__(self, bits=4, pretrained=False):
        super().__init__()
        if pretrained:
            raise AssertionError('Evaluator requested pretrained weights')
        self.hash_layer = nn.Linear(4, bits, bias=False)
        with torch.no_grad():
            self.hash_layer.weight.copy_(torch.eye(bits, 4))

    def forward(self, images):
        return self.hash_layer(images)


def dataset(images, classes, offset):
    labels = F.one_hot(torch.tensor(classes), 100).float()
    ids = torch.arange(offset, offset + len(images))
    data = TensorDataset(torch.tensor(images).float(), labels, ids)
    data.indices = ids.numpy()
    data.labels = np.zeros(offset + len(images), dtype=np.int64)
    data.labels[data.indices] = classes
    return data


class QueryEvaluationTests(unittest.TestCase):
    def test_actual_pixel_fingerprint_tracks_content_and_ignores_array_layout(self):
        from data.data_loader import _dataset_sha256
        images = np.arange(48,dtype=np.uint8).reshape(2,2,4,3)
        labels = np.array([0,1])
        digest = _dataset_sha256(images,labels)
        self.assertEqual(digest,_dataset_sha256(np.asfortranarray(images),labels))
        changed = images.copy()
        changed[0,0,0,0] += 1
        self.assertNotEqual(digest,_dataset_sha256(changed,labels))
        self.assertNotEqual(digest,_dataset_sha256(images,labels[::-1]))

    def fixture(self, folder):
        root = Path(folder)
        run = root/'run'
        run.mkdir()
        raw = root/'raw'
        raw.mkdir()
        for name in ('train', 'test', 'meta'):
            (raw/name).write_bytes(b'synthetic')
        config = dict(variant='b3fcd', dataset='cifar-100-new-seg', num_classes=100,
                      code_length=4, paired_stage_seeds=True, seed=60, epoch=3,
                      topK=[-1,100,1000], source_sha256=source_sha256(),
                      split_hash='synthetic-split', dataset_content_sha256='synthetic-pixels',
                      non_deterministic=False, root=str(raw), run_id='synthetic',
                      batch_size=2, num_workers=0, fcd_mode='neighbor')
        (run/'config.json').write_text(json.dumps(config))
        history = [dict(epoch=i, validation=dict(zip(evaluator.METRICS, values)))
                   for i, values in enumerate(((.4,.9,.5),(.6,.7,.65),(.6,.8,.7)),1)]
        (run/'training_history.json').write_text(json.dumps(history))
        torch.save(TinyStudent().state_dict(), run/'best_model_state.pt')
        metrics = {'selected_epoch':2, 'selection_metric':'validation mAP@ALL',
                   'best_validation_mAP@ALL':.6, 'query_evaluated':False,
                   **selection_receipt(run)}
        (run/'metrics.json').write_text(json.dumps(metrics))
        query = dataset([[1,1,-1,-1],[0,1,1,-1]], [0,1], 0)
        database = dataset([[1,1,-1,-1],[-1,1,1,-1],[1,1,-1,-1],[-1,-1,1,1]],
                           [0,1,2,1], 2)
        query_loader = DataLoader(query,batch_size=2,shuffle=False)
        database_loader = DataLoader(database,batch_size=2,shuffle=False)
        def loaders(args):
            args.split_hash = 'synthetic-split'
            args.dataset_content_sha256 = 'synthetic-pixels'
            split = Path(args.output_dir)/'splits'
            split.mkdir()
            (split/'synthetic.json').write_text('{}')
            return object(),object(),object(),query_loader,database_loader,0,0,2,4
        options = argparse.Namespace(run_dir=str(run), output_dir=None, data_root=None,
                                     cpu=True, gpu=0, dry_run=False)
        return run, options, loaders, query_loader, database_loader

    def invoke(self, options, loaders):
        with patch.object(evaluator,'ResNet',TinyStudent), \
             patch.object(evaluator,'load_data',side_effect=loaders), \
             patch('torch.optim.RMSprop',side_effect=AssertionError('No optimizer allowed')), \
             patch('torch.hub.load',side_effect=AssertionError('No teacher allowed')), \
             patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads allowed')):
            return evaluator.evaluate(options)

    def test_selected_checkpoint_metrics_ids_and_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            run,options,loaders,query,database = self.fixture(folder)
            before = {name:file_sha256(run/name) for name in
                      ('config.json','metrics.json','training_history.json','best_model_state.pt')}
            report = self.invoke(options,loaders)
            self.assertTrue(report['query_evaluated'])
            self.assertEqual(report['selected_epoch'],2)  # First max; not last epoch or @100 peak.
            saved = torch.load(run/'final_query'/'retrieval_outputs.pt',weights_only=False)
            values,_ = CalcTopMapWithPR(saved['query_binary'].numpy(),saved['query_label'].numpy(),
                                       saved['database_binary'].numpy(),saved['database_label'].numpy(),[-1,100,1000])
            self.assertEqual(report['query_metrics'],dict(zip(evaluator.METRICS,values)))
            expected = CalcTopMapPerQuery(saved['query_binary'].numpy(),saved['query_label'].numpy(),
                                          saved['database_binary'].numpy(),saved['database_label'].numpy(),[-1,100,1000])
            for key in expected:
                np.testing.assert_array_equal(saved['per_query_ap'][key],expected[key])
            self.assertEqual(saved['query_ids'].tolist(),[0,1])
            self.assertEqual(saved['database_ids'].tolist(),[2,3,4,5])
            self.assertTrue(((saved['query_binary']==-1)|(saved['query_binary']==1)).all())
            for name,digest in before.items():
                self.assertEqual(file_sha256(run/name),digest)
            self.assertTrue((run/'final_query'/'splits'/'synthetic.json').exists())
            with self.assertRaises(FileExistsError):
                self.invoke(options,loaders)

    def test_dry_run_never_forwards_or_creates_evaluation_output(self):
        with tempfile.TemporaryDirectory() as folder:
            run,options,loaders,*_ = self.fixture(folder)
            options.dry_run = True
            with patch.object(TinyStudent,'forward',side_effect=AssertionError('No query forwards')):
                report = self.invoke(options,loaders)
            self.assertTrue(report['dry_run'])
            self.assertFalse(report['query_evaluated'])
            self.assertFalse((run/'final_query').exists())

    def test_source_config_checkpoint_and_history_corruption_rejected(self):
        for filename in ('config.json','best_model_state.pt','training_history.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as folder:
                run,options,loaders,*_ = self.fixture(folder)
                with (run/filename).open('ab') as stream:
                    stream.write(b' ')
                with self.assertRaisesRegex(ValueError,'receipt mismatch'):
                    self.invoke(options,loaders)
        with tempfile.TemporaryDirectory() as folder:
            run,options,loaders,*_ = self.fixture(folder)
            with patch.object(evaluator,'source_sha256',return_value='wrong-source'):
                with self.assertRaisesRegex(ValueError,'source SHA256'):
                    self.invoke(options,loaders)

    def test_selection_and_incomplete_history_rejected_even_with_updated_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            run,options,loaders,*_ = self.fixture(folder)
            summary = json.loads((run/'metrics.json').read_text())
            summary['selected_epoch'] = 3
            (run/'metrics.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'Selected epoch'):
                self.invoke(options,loaders)
            history = json.loads((run/'training_history.json').read_text())[:-1]
            (run/'training_history.json').write_text(json.dumps(history))
            summary.update(selection_receipt(run))
            (run/'metrics.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                self.invoke(options,loaders)

    def test_split_and_actual_dataset_mismatch_rejected_before_query(self):
        for field in ('split_hash','dataset_content_sha256'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                run,options,loaders,*_ = self.fixture(folder)
                def wrong(args):
                    values = loaders(args)
                    setattr(args,field,'wrong')
                    return values
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                    self.invoke(options,wrong)
                self.assertFalse((run/'final_query').exists())

    def test_strict_state_loading_and_missing_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            run,options,loaders,*_ = self.fixture(folder)
            state = TinyStudent().state_dict()
            state['unexpected_auxiliary_head'] = torch.zeros(1)
            torch.save(state,run/'best_model_state.pt')
            summary = json.loads((run/'metrics.json').read_text())
            summary.update(selection_receipt(run))
            (run/'metrics.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError,'Unexpected key'):
                self.invoke(options,loaders)
            summary.pop('selection_receipt_version')
            (run/'metrics.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'Missing selection receipt'):
                self.invoke(options,loaders)

    def test_resnet_evaluation_constructor_has_identical_state_keys_and_no_download(self):
        import network
        with patch('network._build_resnet',side_effect=AssertionError('Pretrained path forbidden')), \
             patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads')):
            model = network.ResNet(32,pretrained=False)
        self.assertEqual(model.hash_layer.weight.shape,(32,512))
        self.assertIn('feature_layers.0.weight',model.state_dict())
        self.assertIn('conv1.weight',model.state_dict())


if __name__ == '__main__':
    unittest.main()
