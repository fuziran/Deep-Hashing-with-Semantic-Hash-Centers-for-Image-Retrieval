"""Frozen weak-view teacher logits -> training-only classifier on hash codes."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from utils.reliable_relation import tensor_sha256

VERSION = "i3-foundation-discriminative-v1"
CACHE_VERSION = "foundation-teacher-logits-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_subset_sha256(dataset):
    """Fingerprint only the actual training images/labels, not query data."""
    import numpy as np
    ids = np.sort(np.asarray(dataset.indices, dtype=np.int64))
    digest = hashlib.sha256(ids.astype('<i8').tobytes())
    for index in ids:
        digest.update(np.asarray(dataset.images[index]).tobytes())
        digest.update(np.asarray([dataset.labels[index]], dtype='<i8').tobytes())
    return digest.hexdigest()


def ramp(epoch, maximum=.05, warmup=40, end=60):
    if (not isinstance(epoch, int) or epoch < 1 or not math.isfinite(maximum) or maximum < 0
            or not isinstance(warmup,int) or not isinstance(end,int) or not 0 <= warmup < end):
        raise ValueError('Invalid FCD weight/schedule')
    return maximum * min(1., max(0., (epoch-warmup)/(end-warmup)))


def validate_exploratory_diagnostic(cache_path, payload, diagnostic_path, nfd_config=None):
    """Validate independent logit evidence without changing the failed gate."""
    import json
    path = Path(diagnostic_path).resolve()
    report = json.loads(path.read_text(encoding='utf-8'))
    identity = payload['identity']
    if (payload.get('teacher_check', {}).get('passed') is not False
            or report.get('teacher_check_passed') is not False
            or Path(report.get('teacher_dir', '')).resolve() != Path(cache_path).resolve().parent
            or report.get('foundation_weights_sha256') != identity['foundation_weights_sha256']
            or report.get('counts', {}).get('train') != len(payload['ids'])):
        raise ValueError('Exploratory diagnostic does not match the failed teacher cache')
    if nfd_config is not None and Path(report.get('nfd_config', '')).resolve() != Path(nfd_config).resolve():
        raise ValueError('Exploratory diagnostic does not match the NFD reference')
    if report.get('diagnostic_source_sha256') != file_sha256(
            Path(__file__).resolve().parents[1] / 'scripts' / 'diagnose_fcd_logits.py'):
        raise ValueError('Exploratory diagnostic source differs from the current checkout')
    scores = report.get('soft_center_T2', {})
    for metric in ('mAP@ALL', 'mAP@100', 'mAP@1000'):
        item = scores.get(metric, {})
        lower, upper = item.get('paired_bootstrap_95', (None, None))
        values = (item.get('new'), item.get('old'), item.get('delta'), lower, upper)
        if (not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
                or abs(values[0] - values[1] - values[2]) > 1e-6
                or values[2] <= 0 or lower <= 0 or upper < lower):
            raise ValueError(f'Exploratory direct-logit diagnostic lacks positive {metric} evidence')
    return file_sha256(path)


def decoupled_loss(student, teacher, targets, temperature=2., weights=None):
    """Stable TCKD + NCKD; optional equal-policy non-target reweighting.

    Both distributions use the same weights before normalization. True class
    remains excluded from NCKD; semantically near negatives never become labels.
    """
    if (student.ndim != 2 or student.shape != teacher.shape or student.shape[1] < 3
            or targets.shape != (len(student),) or targets.dtype != torch.long
            or (targets<0).any() or (targets>=student.shape[1]).any()
            or not math.isfinite(temperature) or temperature <= 0
            or not torch.isfinite(student).all() or not torch.isfinite(teacher).all()):
        raise ValueError('Invalid DKD inputs')
    mask = F.one_hot(targets, student.shape[1]).bool()
    s, t = student/temperature, teacher.detach()/temperature
    def binary_logits(z):
        return torch.stack([z.gather(1,targets[:,None]).squeeze(1),
                            z.masked_fill(mask,float('-inf')).logsumexp(1)],1)
    binary_s, binary_t = binary_logits(s), binary_logits(t)
    tckd = F.kl_div(F.log_softmax(binary_s,1),F.softmax(binary_t,1),reduction='batchmean')
    ns, nt = s[~mask].reshape(len(s),-1), t[~mask].reshape(len(t),-1)
    if weights is not None:
        if (weights.shape != student.shape or not torch.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError('DKD weights must be finite and positive')
        logw = weights.detach()[~mask].reshape_as(ns).log()
        ns, nt = ns+logw, nt+logw
    nckd = F.kl_div(F.log_softmax(ns,1),F.softmax(nt,1),reduction='batchmean')
    return tckd*temperature**2, nckd*temperature**2


class FoundationDistillation(nn.Module):
    def __init__(self, centers, similarity, ids, logits, labels, mode='neighbor',
                 temperature=2., gamma=1., topk=5):
        super().__init__()
        classes, bits = centers.shape
        if (mode not in ('ce','plain','neighbor') or classes < 3
                or similarity.shape != (classes,classes) or not torch.isfinite(similarity).all()
                or not ((centers == -1) | (centers == 1)).all()
                or ids.dtype != torch.long or ids.ndim != 1 or len(ids) == 0
                or logits.shape != (len(ids),classes) or labels.shape != ids.shape
                or labels.dtype != torch.long or not torch.isfinite(logits).all()
                or (labels<0).any() or (labels>=classes).any()
                or len(ids.unique()) != len(ids) or (ids[1:] <= ids[:-1]).any()
                or not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(gamma) or not 0 <= gamma <= 1
                or not isinstance(topk,int) or not 0 < topk < classes):
            raise ValueError('Invalid FCD centers/cache/options')
        # Preserve paired initialization/RNG, including in zero-weight controls.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.Linear(bits,classes)
        with torch.no_grad():
            self.head.weight.copy_(centers.cpu().float()*4/bits)
            self.head.bias.zero_()
        self.register_buffer('ids',ids.detach().cpu().clone())
        self.register_buffer('teacher_logits',logits.detach().cpu().float().clone())
        self.register_buffer('teacher_labels',labels.detach().cpu().clone())
        weights = torch.ones(classes,classes)
        for y in range(classes):
            near=sorted((c for c in range(classes) if c!=y),key=lambda c:(-float(similarity[y,c]),c))[:topk]
            if mode=='neighbor':
                weights[y,near] += gamma*similarity[y,near].detach().cpu().clamp(0,1)
        self.register_buffer('negative_weights',weights)
        self.mode,self.temperature=mode,temperature
        self.components={}

    def forward(self, codes, targets, image_ids):
        ids=image_ids.to(self.ids.device,dtype=torch.long)
        positions=torch.searchsorted(self.ids,ids)
        if (positions>=len(self.ids)).any() or not torch.equal(self.ids[positions],ids):
            raise ValueError('FCD requested an image outside the training cache')
        if not torch.equal(self.teacher_labels[positions],targets):
            raise ValueError('FCD cached label does not match training label')
        prediction=self.head(codes)
        ce=F.cross_entropy(prediction,targets)
        if self.mode=='ce':
            tckd,nckd=ce.new_zeros(()),ce.new_zeros(())
        else:
            tckd,nckd=decoupled_loss(prediction,self.teacher_logits[positions],targets,self.temperature,
                                    self.negative_weights[targets] if self.mode=='neighbor' else None)
        loss=ce+tckd+nckd
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite FCD loss')
        self.components=dict(loss_fcd=loss.detach().item(),fcd_ce=ce.detach().item(),
                             fcd_tckd=tckd.detach().item(),fcd_nckd=nckd.detach().item(),
                             fcd_neighbor_weight_mean=self.negative_weights[targets].mean().item(),
                             fcd_student_aux_accuracy=(prediction.detach().argmax(1)==targets).float().mean().item())
        return loss


def load_fcd(args, centers, similarity, train_dataset):
    if args.fcd_weight == 0:
        args.fcd_disabled=True
        return None
    payload=torch.load(args.fcd_cache,map_location='cpu',weights_only=False)
    identity=payload['identity']
    expected=dict(version=CACHE_VERSION,dataset=args.dataset,seed=args.seed,
                  split_hash=args.split_hash,num_classes=args.num_classes,
                  classifier_sha256=args.classifier_sha256,resize_size=args.resize_size,
                  crop_size=args.crop_size,training_images_sha256=image_subset_sha256(train_dataset))
    for key,value in expected.items():
        if identity.get(key)!=value: raise ValueError(f'Foundation cache mismatch: {key}')
    for name in ('ids','logits','labels'):
        if tensor_sha256(payload[name])!=identity.get(name+'_sha256'):
            raise ValueError(f'Foundation cache content mismatch: {name}')
    import numpy as np
    if not np.array_equal(payload['ids'].numpy(),np.sort(train_dataset.indices)):
        raise ValueError('Foundation cache training IDs differ')
    if not np.array_equal(payload['labels'].numpy(),np.asarray(train_dataset.labels)[payload['ids'].numpy()]):
        raise ValueError('Foundation cache labels differ')
    passed = payload.get('teacher_check',{}).get('passed') is True
    diagnostic = getattr(args, 'fcd_exploratory_diagnostic', None)
    if not passed:
        if not diagnostic:
            raise ValueError('Foundation teacher failed or lacks validation-only information check; inspect teacher_report.json')
        args.fcd_diagnostic_sha256 = validate_exploratory_diagnostic(args.fcd_cache,payload,diagnostic)
        args.fcd_exploratory = True
        args.fcd_original_teacher_check_passed = False
    elif diagnostic:
        raise ValueError('Exploratory diagnostic is only for a teacher with a failed original gate')
    else:
        args.fcd_exploratory = False
        args.fcd_original_teacher_check_passed = True
    if centers.shape != (args.num_classes,args.code_length): centers=centers.T
    module=FoundationDistillation(centers,similarity,payload['ids'],payload['logits'],payload['labels'],
                                  args.fcd_mode,args.fcd_temperature,args.fcd_gamma,args.fcd_topk)
    args.fcd_cache_sha256=file_sha256(args.fcd_cache)
    args.fcd_foundation_weights_sha256=identity['foundation_weights_sha256']
    args.fcd_teacher_identity=identity
    # Manifest writer accepts scalar/list fields; keep full identity in a string.
    import json
    args.fcd_teacher_identity_json=json.dumps(identity,sort_keys=True)
    args.fcd_view_policy='cached-center-and-flip-weak-teacher-to-original-random-crop-student'
    return module.to(args.device)
