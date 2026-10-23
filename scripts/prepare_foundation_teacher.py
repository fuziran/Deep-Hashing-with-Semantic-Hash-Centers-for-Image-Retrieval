"""Offline DINO teacher preparation. Does not evaluate final query or database.

Use a separate recent-PyTorch environment for DINOv3. Export plain CPU tensors
usable by the unchanged SHC training environment. Load only trusted local weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

import torch
from torch import nn
from torch.nn import functional as F

from utils.foundation_distill import CACHE_VERSION, file_sha256, image_subset_sha256
from utils.reliable_relation import tensor_sha256


@torch.no_grad()
def extract(loader, foundation, old_teacher, device):
    rows=[]
    foundation.eval();old_teacher.eval()
    for batch,(images,labels,ids) in enumerate(loader):
        images=images.to(device)
        views=[images,images.flip(-1)]
        new_features=[];old_features=[];old_logits=[]
        for view in views:
            f=foundation(view)
            if not isinstance(f,torch.Tensor) or f.ndim!=2:
                raise ValueError('Foundation backbone must return [batch,features] embeddings')
            if not torch.isfinite(f).all() or (f.norm(dim=1)<=1e-12).any():
                raise ValueError('Invalid foundation feature')
            logits,old=old_teacher.forward_with_features(view)
            new_features.append(F.normalize(f.float(),dim=1))
            old_features.append(F.normalize(old.float(),dim=1));old_logits.append(logits.float())
        rows.append((ids.cpu(),labels.argmax(1).cpu(),
                     F.normalize(sum(new_features)/2,dim=1).cpu(),
                     F.normalize(sum(old_features)/2,dim=1).cpu(),
                     (sum(old_logits)/2).cpu()))
        if (batch+1)%25==0: print(f'Teacher extraction: batch {batch+1}/{len(loader)}',flush=True)
    values=[torch.cat([row[i] for row in rows]) for i in range(5)]
    order=values[0].argsort()
    return [value[order] for value in values]


def fit_classifier(train_features, labels, validation_features, validation_labels, classes,
                   device, steps=200):
    head=nn.Linear(train_features.shape[1],classes).to(device)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.001)
    x,y,v,vy=[value.to(device) for value in (train_features,labels,validation_features,validation_labels)]
    best_accuracy=-1.;best=None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(head(x),y);loss.backward();optimizer.step()
        if (step+1)%10==0 or step+1==steps:
            with torch.no_grad(): accuracy=(head(v).argmax(1)==vy).float().mean().item()
            if accuracy>best_accuracy:
                best_accuracy=accuracy
                best={k:value.detach().clone() for k,value in head.state_dict().items()}
    head.load_state_dict(best)
    with torch.no_grad(): train_logits=head(x).cpu();validation_logits=head(v).cpu()
    return head.cpu(),train_logits,validation_logits,best_accuracy


def fit_hash_probe(features,labels,validation_features,centers,device,steps=200):
    """Fixed-budget frozen-feature linear probe; no validation optimization."""
    head=nn.Linear(features.shape[1],centers.shape[1]).to(device)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.001)
    x=features.to(device);target=(centers[labels].to(device)+1)/2
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(2*head(x),target)
        loss.backward();optimizer.step()
    with torch.no_grad():
        train_codes=torch.where(head(x)>=0,1.,-1.).cpu()
        validation_codes=torch.where(head(validation_features.to(device))>=0,1.,-1.).cpu()
    return train_codes,validation_codes


def retrieval_metrics(train_codes,validation_codes,train_labels,validation_labels,classes):
    from utils.tools import CalcTopMapPerQuery
    values=CalcTopMapPerQuery(validation_codes.numpy(),F.one_hot(validation_labels,classes).numpy(),
                              train_codes.numpy(),F.one_hot(train_labels,classes).numpy(),[-1,100,1000])
    return {name:float(values[key].mean()) for name,key in
            [('mAP@ALL','-1'),('mAP@100','100'),('mAP@1000','1000')]}


def teacher_check(new_logits,old_logits,labels,new_proxy,old_proxy):
    new_correct=new_logits.argmax(1)==labels
    old_correct=old_logits.argmax(1)==labels
    result=dict(new_accuracy=new_correct.float().mean().item(),old_accuracy=old_correct.float().mean().item(),
                corrected=int((new_correct & ~old_correct).sum()),regressed=int((~new_correct & old_correct).sum()),
                new_hash_probe=new_proxy,old_hash_probe=old_proxy,
                retrieval_protocol='validation vs training-only gallery; fixed 32-bit B1 centers; diagnostic only')
    result['passed']=(result['new_accuracy']>result['old_accuracy'] and result['corrected']>result['regressed']
                      and new_proxy['mAP@ALL']>old_proxy['mAP@ALL']
                      and all(new_proxy[k]>=old_proxy[k] for k in ('mAP@100','mAP@1000')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nfd-config',required=True,help='Existing audited 32-bit NFD config')
    parser.add_argument('--foundation-repo',required=True,help='Local official DINOv3 or DINOv2 checkout')
    parser.add_argument('--foundation-model',choices=('dinov3_vitb16','dinov2_vitb14_reg'),default='dinov3_vitb16')
    parser.add_argument('--foundation-weights',required=True,help='Local official checkpoint; no automatic download')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--probe-steps',type=int,default=200)
    parser.add_argument('--extraction-batch-size',type=int,default=32)
    options=parser.parse_args()
    if options.foundation_model.startswith('dinov3'):
        import re
        version=tuple(map(int,re.match(r'(\d+)\.(\d+)\.(\d+)',str(torch.__version__)).groups()))
        if version < (2,7,1):
            parser.error('DINOv3 preparation requires a separate PyTorch >= 2.7.1 environment; keep original SHC environment unchanged')
    root=Path(options.output_dir).resolve()
    if root.exists(): parser.error('Teacher output already exists; choose a new directory')
    if options.probe_steps<1 or options.extraction_batch_size<1:
        parser.error('probe-steps and extraction-batch-size must be positive')
    model_repo=Path(options.foundation_repo).resolve();weights=Path(options.foundation_weights).resolve()
    if not (model_repo/'hubconf.py').is_file() or not weights.is_file():
        parser.error('Supply existing local official model repository and downloaded weights')
    cfg=json.loads(Path(options.nfd_config).read_text(encoding='utf-8'))
    if (cfg.get('variant')!='b2nfd' or cfg.get('code_length')!=32
            or cfg.get('dataset')!='cifar-100-new-seg' or not cfg.get('paired_stage_seeds')):
        parser.error('Preparation requires the paired CIFAR-100 32-bit NFD reference')
    from scripts.run_i3_npa import ARTIFACTS,resolve_file
    reference=Path(options.nfd_config).resolve()
    for key in ARTIFACTS: cfg[key]=str(resolve_file(cfg[key],reference))
    device=torch.device('cpu' if options.cpu else f'cuda:{options.gpu}')
    if device.type=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    args=argparse.Namespace(**cfg);args.device=device;args.output_dir=str(root)
    from utils.experiment import seed_everything
    seed_everything(args.seed,deterministic=not args.non_deterministic)
    # Model creation uses official local hub API; pretrained=False prevents any
    # unrequested URL fetch. Checkpoint architecture is validated strictly.
    foundation=torch.hub.load(str(model_repo),options.foundation_model,source='local',pretrained=False)
    state=torch.load(weights,map_location='cpu',weights_only=False)
    foundation.load_state_dict(state,strict=True)
    del state
    foundation=foundation.eval().requires_grad_(False).to(device)
    from data.data_loader import load_data
    train,relation,val,_,_,*_=load_data(args)
    if (set(map(int,train.dataset.indices)) != set(map(int,relation.dataset.indices))
            or set(map(int,train.dataset.indices)) & set(map(int,val.dataset.indices))):
        raise ValueError('Teacher preparation requires exact training relation IDs and disjoint validation IDs')
    from utils.frozen_b1 import load_frozen_b1
    from utils.neighbor_feature_distill import load_nfd
    _,centers=load_frozen_b1(args)
    nfd=load_nfd(args)
    from torch.utils.data import DataLoader
    def extraction_loader(dataset):
        return DataLoader(dataset,batch_size=options.extraction_batch_size,shuffle=False,
                          num_workers=args.num_workers,pin_memory=device.type=='cuda')
    relation=extraction_loader(relation.dataset)
    val=extraction_loader(val.dataset)
    print('Extracting training-only and validation-only teacher views; final query/database are not iterated',flush=True)
    train_ids,train_labels,new_train,old_train,_=extract(relation,foundation,nfd.teacher,device)
    _,validation_labels,new_val,old_val,old_logits=extract(val,foundation,nfd.teacher,device)
    # Student-training cache must contain exactly one row per training ID.
    import numpy as np
    if not np.array_equal(train_ids.numpy(),np.sort(train.dataset.indices)):
        raise ValueError('Teacher extraction training IDs differ')
    del foundation,nfd
    if device.type=='cuda': torch.cuda.empty_cache()
    head,logits,new_val_logits,accuracy=fit_classifier(new_train,train_labels,new_val,validation_labels,
                                                     args.num_classes,device,options.probe_steps)
    if centers.shape!=(args.num_classes,32): centers=centers.T
    new_codes,new_vcodes=fit_hash_probe(new_train,train_labels,new_val,centers,device,options.probe_steps)
    old_codes,old_vcodes=fit_hash_probe(old_train,train_labels,old_val,centers,device,options.probe_steps)
    new_proxy=retrieval_metrics(new_codes,new_vcodes,train_labels,validation_labels,args.num_classes)
    old_proxy=retrieval_metrics(old_codes,old_vcodes,train_labels,validation_labels,args.num_classes)
    report=teacher_check(new_val_logits,old_logits,validation_labels,new_proxy,old_proxy)
    report.update(probe_steps=options.probe_steps,classification_selection='best validation accuracy every 10 steps',
                  hash_probe_selection='fixed final step',query_evaluated=False)
    try: repo_commit=subprocess.check_output(['git','-C',str(model_repo),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): repo_commit='unknown'
    from utils.reliable_relation import state_sha256
    identity=dict(version=CACHE_VERSION,dataset=args.dataset,seed=args.seed,split_hash=args.split_hash,
                  num_classes=args.num_classes,classifier_sha256=args.classifier_sha256,
                  resize_size=args.resize_size,crop_size=args.crop_size,
                  training_images_sha256=image_subset_sha256(train.dataset),
                  foundation_model=options.foundation_model,foundation_weights_sha256=file_sha256(weights),
                  foundation_repo_commit=repo_commit,teacher_head_sha256=state_sha256(head.state_dict()),
                  view_policy='normalize-each-center-and-flip-then-normalize-mean',
                  probe_steps=options.probe_steps,preparation_torch_version=str(torch.__version__),
                  preparation_source_sha256=file_sha256(__file__),
                  extraction_batch_size=options.extraction_batch_size)
    for name,value in [('ids',train_ids),('labels',train_labels),('logits',logits)]:
        identity[name+'_sha256']=tensor_sha256(value)
    payload=dict(identity=identity,ids=train_ids,labels=train_labels,logits=logits,teacher_check=report)
    torch.save(payload,root/'foundation_cache.pt')
    torch.save(head.state_dict(),root/'foundation_classifier.pt')
    (root/'teacher_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print(f'Teacher cache: {root / "foundation_cache.pt"}',flush=True)
    if not report['passed']:
        raise SystemExit('Teacher information check did not pass; hash training is intentionally gated. Inspect report before further experiments.')


if __name__=='__main__': main()
