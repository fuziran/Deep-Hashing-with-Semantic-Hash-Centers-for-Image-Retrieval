"""Launch independent 16/32/64-bit I3 runs using each audited NFD configuration.

Run from the repository root. The launcher never loads a parent's student model,
never selects a config by retrieval score, and does not guess missing cache paths.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SHARED = (
    "dataset", "seed", "num_classes", "code_length", "lr", "epoch", "classify_epoch",
    "test_map", "beta", "lambd", "batch_size", "num_workers", "resize_size", "crop_size",
    "val_per_class", "mask_strategy", "center_update_strategy", "relation_mode",
    "relation_kappa", "relation_temperature", "relation_weight_floor", "nfd_alpha",
    "nfd_weight", "nfd_topk", "nfd_warmup", "nfd_ramp_end",
)
ARTIFACTS = ("parent_b1_config", "similarity_cache", "center_cache", "teacher_cache")


def read_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_file(value, config):
    if not isinstance(value,str) or not value:
        raise ValueError(f"Missing artifact path in {config}")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [REPO/raw, config.parent/raw]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError(f"Artifact is missing or ambiguous: {value} (from {config}); fix the saved path")
    return found.pop()


def select_configs(root, explicit, bits, dataset, seed):
    paths = [Path(p).resolve() for p in explicit] if explicit else sorted(Path(root).resolve().rglob("config.json"))
    selected = {}
    for path in paths:
        try:
            config=read_config(path)
        except (OSError,ValueError) as error:
            if explicit: raise ValueError(f"Cannot read {path}: {error}") from error
            continue
        if (config.get("variant") != "b2nfd" or config.get("dataset") != dataset
                or config.get("seed") != seed or config.get("code_length") not in bits):
            if explicit: raise ValueError(f"Not a matching B2NFD baseline: {path}")
            continue
        bit=config["code_length"]
        if bit in selected:
            raise ValueError(f"Multiple {bit}-bit NFD configs found:\n{selected[bit][0]}\n{path}\n"
                             "Pass exactly one --nfd-config per requested bit length.")
        selected[bit]=(path,config)
    missing=set(bits)-set(selected)
    if missing: raise ValueError(f"Missing NFD configs for bits {sorted(missing)} under {root}")
    return selected


def build_command(path, config, options):
    if config.get("paired_stage_seeds") is not True:
        raise ValueError(f"NFD baseline must use paired-stage-seeds: {path}")
    for key in SHARED + ARTIFACTS + ("root", "topK", "non_deterministic"):
        if key not in config: raise ValueError(f"NFD config missing {key}: {path}")
    if config['topK'] != [-1,100,1000]: raise ValueError(f"Unexpected NFD topK: {path}")
    output=Path(options.output_root).resolve()/f"{config['code_length']}bit_{options.mode}"
    if output.exists():
        raise ValueError(f"Output already exists; choose a new --output-root: {output}")
    command=[sys.executable,str(REPO/'run.py'),'--variant','b3npa','--paired-stage-seeds']
    for key in SHARED:
        command += ['--'+key.replace('_','-'),str(config[key])]
    for key in ARTIFACTS:
        command += ['--'+key.replace('_','-'),str(resolve_file(config[key],path))]
    data_root=options.data_root or config['root']
    data_root=Path(data_root)
    if not data_root.is_absolute(): data_root=REPO/data_root
    if not data_root.is_dir(): raise ValueError(f"Dataset root does not exist: {data_root}")
    if config['non_deterministic']: command+=['--non-deterministic']
    command += ['--root',str(data_root.resolve()),'--topK','-1','100','1000','--gpu',str(options.gpu),
                '--npa-mode',options.mode,'--npa-rho',str(options.rho),
                '--npa-shrinkage',str(options.shrinkage),'--npa-weight',str(options.weight),
                '--npa-warmup',str(options.warmup),'--npa-ramp-end',str(options.ramp_end),
                '--output-dir',str(output)]
    if not options.evaluate_query: command+=['--skip-query']
    return command,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',default='./save')
    parser.add_argument('--nfd-config',action='append',default=[],help='Repeat once per bit if auto-discovery is ambiguous')
    parser.add_argument('--bits',nargs='+',type=int,default=[16,32,64])
    parser.add_argument('--dataset',default='cifar-100-new-seg')
    parser.add_argument('--seed',type=int,default=60)
    parser.add_argument('--data-root',default=None)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--mode',choices=['protected','covariance','isotropic'],default='protected')
    parser.add_argument('--rho',type=float,default=.05)
    parser.add_argument('--shrinkage',type=float,default=.2)
    parser.add_argument('--weight',type=float,default=1.)
    parser.add_argument('--warmup',type=int,default=40)
    parser.add_argument('--ramp-end',type=int,default=60)
    parser.add_argument('--evaluate-query',action='store_true',help='Evaluate final query once after validation-only model selection')
    parser.add_argument('--dry-run',action='store_true',help='Validate paths and print all commands without training')
    options=parser.parse_args()
    if (len(set(options.bits))!=len(options.bits) or any(b not in (16,32,64) for b in options.bits)
            or not math.isfinite(options.rho) or options.rho<0
            or not math.isfinite(options.shrinkage) or not 0<=options.shrinkage<=1
            or not math.isfinite(options.weight) or not 0<=options.weight<=1
            or not 0<=options.warmup<options.ramp_end):
        parser.error('Invalid bits, NPA strength, shrinkage or schedule')
    root=Path(options.output_root).resolve()
    if root.exists(): parser.error(f'Output root already exists; choose a new directory: {root}')
    try:
        configs=select_configs(options.baseline_root,options.nfd_config,options.bits,options.dataset,options.seed)
        plan=[]
        for bit in options.bits:
            path,config=configs[bit]
            if options.weight>0 and options.rho>0 and config['epoch']<options.ramp_end:
                raise ValueError(f'{bit}-bit training budget does not cover NPA ramp')
            command,output=build_command(path,config,options)
            plan.append(dict(bits=bit,nfd_config=str(path),command=command,output=str(output)))
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    for row in plan:
        print(f"\n{row['bits']} bit; inherited NFD config: {row['nfd_config']}",flush=True)
        print(shlex.join(row['command']),flush=True)
    if options.dry_run: return
    root.mkdir(parents=True,exist_ok=False)
    (root/'launch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for row in plan:
        print(f"\nStarting {row['bits']}-bit training; log: {root / (str(row['bits'])+'bit.log')}",flush=True)
        with (root/f"{row['bits']}bit.log").open('w',encoding='utf-8') as log:
            subprocess.run(row['command'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PYTHONUNBUFFERED':'1'})
        runs=list((Path(row['output'])/'runs').glob('*'))
        if len(runs)!=1: raise RuntimeError(f"Expected one run under {row['output']}")
        metrics=read_config(runs[0]/'metrics.json')
        record=dict(bits=row['bits'],run_dir=str(runs[0]),query_evaluated=metrics.get('query_evaluated',True),
                    selected_epoch=metrics['selected_epoch'])
        query=metrics.get('query_metrics') or {}
        record.update({key:query.get(key) for key in ['mAP@ALL','mAP@100','mAP@1000']})
        record['best_validation_mAP@ALL']=metrics['best_validation_mAP@ALL']
        results.append(record)
        with (root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader();writer.writerows(results)
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
