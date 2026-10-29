"""Explicit NFD references -> gated FCD experiments, sequentially on one GPU."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_i3_npa import SHARED, ARTIFACTS, resolve_file, read_config, select_configs


def build_command(path,config,options):
    if not config.get('paired_stage_seeds') or config.get('nfd_alpha',0)<=0 or config.get('nfd_weight',0)<=0:
        raise ValueError(f'Reference must be actual paired NFD, not ordinary/disabled distillation: {path}')
    for key in SHARED+ARTIFACTS+('root','topK','non_deterministic'):
        if key not in config: raise ValueError(f'Reference missing {key}: {path}')
    if config['topK']!=[-1,100,1000]: raise ValueError('Unexpected reference cutoffs')
    output=Path(options.output_root).resolve()/f"{config['code_length']}bit_{options.mode}"
    command=[sys.executable,str(REPO/'run.py'),'--variant','b3fcd','--paired-stage-seeds']
    for key in SHARED: command+=['--'+key.replace('_','-'),str(config[key])]
    for key in ARTIFACTS: command+=['--'+key.replace('_','-'),str(resolve_file(config[key],path))]
    data_root=Path(options.data_root or config['root'])
    if not data_root.is_absolute(): data_root=REPO/data_root
    if not data_root.is_dir(): raise ValueError(f'Data root does not exist: {data_root}')
    command+=['--root',str(data_root.resolve()),'--gpu',str(options.gpu),'--topK','-1','100','1000',
              '--output-dir',str(output),'--fcd-mode',options.mode,'--fcd-weight',str(options.weight),
              '--fcd-kd-scale',str(options.kd_scale),
              '--fcd-temperature',str(options.temperature),'--fcd-gamma',str(options.gamma),
              '--fcd-topk',str(options.topk),'--fcd-warmup',str(options.warmup),'--fcd-ramp-end',str(options.ramp_end)]
    if options.weight>0: command+=['--fcd-cache',str(Path(options.teacher_cache).resolve())]
    if options.exploratory_diagnostic:
        command+=['--fcd-exploratory-diagnostic',str(Path(options.exploratory_diagnostic).resolve())]
    if config['non_deterministic']: command+=['--non-deterministic']
    if not options.evaluate_query: command+=['--skip-query']
    if options.weight>0 and config['epoch']<options.ramp_end: raise ValueError('Training budget does not cover FCD ramp')
    return command,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nfd-config',action='append',required=True)
    parser.add_argument('--bits',nargs='+',type=int,default=[16,32,64])
    parser.add_argument('--dataset',default='cifar-100-new-seg')
    parser.add_argument('--seed',type=int,default=60)
    parser.add_argument('--teacher-cache')
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--data-root')
    parser.add_argument('--mode',choices=('ce','plain','neighbor'),default='neighbor')
    parser.add_argument('--weight',type=float,default=.05)
    parser.add_argument('--kd-scale',type=float,default=1.,
                        help='Scale teacher DKD only; keep auxiliary CE and outer weight fixed')
    parser.add_argument('--temperature',type=float,default=2.)
    parser.add_argument('--gamma',type=float,default=1.)
    parser.add_argument('--topk',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=40)
    parser.add_argument('--ramp-end',type=int,default=60)
    parser.add_argument('--evaluate-query',action='store_true')
    parser.add_argument('--exploratory-diagnostic',help='Direct-logit report for a validation-only pilot after the original teacher gate failed')
    parser.add_argument('--dry-run',action='store_true')
    options=parser.parse_args()
    if (not options.bits or len(set(options.bits))!=len(options.bits) or any(b not in (16,32,64) for b in options.bits)
            or not all(math.isfinite(v) for v in (options.weight,options.kd_scale,
                                                  options.temperature,options.gamma))
            or options.weight<0 or options.temperature<=0 or not 0<=options.gamma<=1
            or not 0<=options.kd_scale<=1
            or not 0<=options.warmup<options.ramp_end or options.topk<1):
        parser.error('Invalid FCD options')
    root=Path(options.output_root).resolve()
    if root.exists(): parser.error(f'Output root already exists: {root}')
    if options.weight>0:
        if not options.teacher_cache or not Path(options.teacher_cache).is_file():
            parser.error('Prepare a foundation_cache.pt first using prepare_foundation_teacher.py')
        import torch
        from utils.foundation_distill import validate_exploratory_diagnostic
        cache=torch.load(options.teacher_cache,map_location='cpu',weights_only=False)
        if cache.get('teacher_check',{}).get('passed') is not True:
            if not options.exploratory_diagnostic:
                parser.error('Teacher information check failed; inspect teacher_report.json')
            if options.evaluate_query:
                parser.error('Exploratory pilot is validation-only; omit --evaluate-query')
            try:
                for reference in options.nfd_config:
                    validate_exploratory_diagnostic(options.teacher_cache,cache,
                                                    options.exploratory_diagnostic,reference)
            except (OSError,ValueError,KeyError,TypeError) as error:
                parser.error(str(error))
        elif options.exploratory_diagnostic:
            parser.error('Exploratory diagnostic is only for a failed original teacher gate')
    elif options.exploratory_diagnostic:
        parser.error('Exploratory diagnostic requires a positive FCD weight')
    try:
        selected=select_configs('.',options.nfd_config,options.bits,options.dataset,options.seed)
        plan=[]
        for bit in options.bits:
            path,config=selected[bit]
            command,output=build_command(path,config,options)
            plan.append(dict(bits=bit,nfd_config=str(path),command=command,output=str(output)))
    except (OSError,ValueError,KeyError) as error: parser.error(str(error))
    for row in plan:
        print(f"{row['bits']} bit: {row['nfd_config']}\n{shlex.join(row['command'])}",flush=True)
    if options.dry_run: return
    root.mkdir(parents=True,exist_ok=False)
    (root/'launch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for row in plan:
        print(f"Starting {row['bits']} bit. Log: {root / (str(row['bits'])+'bit.log')}",flush=True)
        with (root/f"{row['bits']}bit.log").open('w',encoding='utf-8') as log:
            subprocess.run(row['command'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PYTHONUNBUFFERED':'1'})
        runs=list((Path(row['output'])/'runs').glob('*'))
        if len(runs)!=1: raise RuntimeError(f'Expected one run: {row["output"]}')
        metrics=read_config(runs[0]/'metrics.json')
        record=dict(bits=row['bits'],run_dir=str(runs[0]),selected_epoch=metrics['selected_epoch'],
                    query_evaluated=metrics['query_evaluated'],
                    fcd_exploratory=metrics.get('fcd_exploratory',False),
                    best_validation_mAP_ALL=metrics['best_validation_mAP@ALL'])
        record.update({key:(metrics.get('query_metrics') or {}).get(key) for key in ('mAP@ALL','mAP@100','mAP@1000')})
        results.append(record)
        with (root/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record));writer.writeheader();writer.writerows(results)
        print(json.dumps(record),flush=True)


if __name__=='__main__': main()
