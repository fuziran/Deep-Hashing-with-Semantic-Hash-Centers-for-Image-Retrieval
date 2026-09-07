"""Run metadata respects the selected device, independently of host CUDA support."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils.experiment import write_run_manifest


class ManifestDeviceTests(unittest.TestCase):
    def test_cpu_on_cuda_host_does_not_query_gpu(self):
        for device in (torch.device('cpu'), 'cpu'):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(run_dir=directory, device=device, gpu=0)
                with patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.get_device_name', side_effect=AssertionError('CPU is not CUDA')) as name:
                    write_run_manifest(args)
                name.assert_not_called()
                saved = json.loads((Path(directory)/'config.json').read_text())
                self.assertEqual(saved['device'], 'cpu')
                self.assertTrue(saved['cuda_available'])
                self.assertEqual(saved['gpu'], 0)  # Retain the CLI option, not a queried name.

    def test_cuda_records_selected_gpu_name(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(run_dir=directory, device=torch.device('cuda:1'), gpu=1)
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.get_device_name', return_value='Synthetic GPU') as name:
                write_run_manifest(args)
            name.assert_called_once_with(args.device)
            saved = json.loads((Path(directory)/'config.json').read_text())
            self.assertEqual(saved['device'], 'cuda:1')
            self.assertEqual(saved['gpu'], 'Synthetic GPU')

    def test_cpu_on_cpu_host_preserves_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(run_dir=directory, device=torch.device('cpu'))
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.cuda.get_device_name') as name:
                write_run_manifest(args)
            name.assert_not_called()
            saved = json.loads((Path(directory)/'config.json').read_text())
            self.assertFalse(saved['cuda_available'])
            self.assertNotIn('gpu', saved)


if __name__ == '__main__':
    unittest.main()
