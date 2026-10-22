"""Launcher preflight and inheritance without starting real training."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_i3_npa as launch


class LauncherTests(unittest.TestCase):
    def fixture(self,root,bit=32):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        config={key:1 for key in launch.SHARED}
        config.update(variant='b2nfd',dataset='cifar-100-new-seg',seed=60,code_length=bit,
                      epoch=300,num_workers=4,paired_stage_seeds=True,topK=[-1,100,1000],
                      non_deterministic=False,root=str(root),nfd_alpha=.75,nfd_weight=.01,
                      nfd_warmup=20,nfd_ramp_end=40)
        for key in launch.ARTIFACTS:
            file=root/(key+'.pt');file.write_text('fixture');config[key]=str(file)
        path=root/'config.json';path.write_text(json.dumps(config))
        return path,config

    def options(self,root,**kwargs):
        values=dict(output_root=str(Path(root)/'output'),mode='protected',data_root=None,gpu=0,
                    rho=.05,shrinkage=.2,weight=1.,warmup=40,ramp_end=60,evaluate_query=False)
        values.update(kwargs);return argparse.Namespace(**values)

    def test_inherit_protocol_nfd_paths_and_validation_only_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path,config=self.fixture(Path(folder)/'base')
            command,_=launch.build_command(path,config,self.options(folder))
            for field in ['epoch','nfd-alpha','nfd-weight','num-workers','code-length']:
                self.assertEqual(command[command.index('--'+field)+1],str(config[field.replace('-','_')]))
            self.assertIn('--skip-query',command)
            self.assertEqual(command[command.index('--variant')+1],'b3npa')
            self.assertNotIn('best_model_state.pt',' '.join(command))
            formal,_=launch.build_command(path,config,self.options(folder,evaluate_query=True))
            self.assertNotIn('--skip-query',formal)

    def test_ambiguous_configs_fail_and_explicit_selection_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            first,_=self.fixture(Path(folder)/'one')
            self.fixture(Path(folder)/'two')
            with self.assertRaisesRegex(ValueError,'Multiple'): launch.select_configs(folder,[],[32],'cifar-100-new-seg',60)
            chosen=launch.select_configs(folder,[str(first)],[32],'cifar-100-new-seg',60)
            self.assertEqual(chosen[32][0],first)
            with self.assertRaisesRegex(ValueError,'Missing NFD'): launch.select_configs(folder,[str(first)],[16,32],'cifar-100-new-seg',60)

    def test_missing_artifacts_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,config=self.fixture(Path(folder)/'base')
            bad=dict(config,teacher_cache=str(Path(folder)/'missing.pt'))
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'): launch.build_command(path,bad,self.options(folder))
            output=Path(folder)/'output'/'32bit_protected';output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'Output already exists'): launch.build_command(path,config,self.options(folder))

    def test_dry_run_preflights_all_lengths_without_launch_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            for bit in [16,32,64]: self.fixture(Path(folder)/str(bit),bit)
            out=Path(folder)/'out'
            with patch('sys.argv',['launcher','--baseline-root',folder,'--output-root',str(out),'--dry-run']), \
                 patch('subprocess.run') as run,patch('builtins.print'):
                launch.main()
            run.assert_not_called();self.assertFalse(out.exists())

    def test_three_runs_summary_uses_selected_final_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            for bit in [16,32,64]: self.fixture(Path(folder)/str(bit),bit)
            out=Path(folder)/'out'
            def fake_run(command,**kwargs):
                run_dir=Path(command[command.index('--output-dir')+1])/'runs'/'synthetic'
                run_dir.mkdir(parents=True)
                (run_dir/'metrics.json').write_text(json.dumps({'selected_epoch':10,
                    'best_validation_mAP@ALL':.5,'query_evaluated':True,
                    'query_metrics':{'mAP@ALL':.6,'mAP@100':.7,'mAP@1000':.65}}))
            with patch('sys.argv',['launcher','--baseline-root',folder,'--output-root',str(out),'--evaluate-query']), \
                 patch('subprocess.run',side_effect=fake_run) as run,patch('builtins.print'):
                launch.main()
            self.assertEqual(run.call_count,3)
            import csv
            with (out/'summary.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['bits'] for r in rows],['16','32','64'])
            self.assertTrue(all(r['mAP@ALL']=='0.6' for r in rows))


if __name__=='__main__': unittest.main()
