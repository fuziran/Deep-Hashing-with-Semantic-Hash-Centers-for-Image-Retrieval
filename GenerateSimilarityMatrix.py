import os.path
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from utils.tools import *
from network import *
from loguru import logger

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
_CLIP_PROMPT_TEMPLATE = "a photo of a {}."


def get_similarity_matrix_path(args):
    os.makedirs('./save/SimilarityMatrix/', exist_ok=True)
    tag = '_AdaptiveFusion' if getattr(args, 'use_adaptive_fusion', False) else ''
    return f'./save/SimilarityMatrix/{args.dataset}{tag}_Similarity_Matrix.pt'


def TrainClassificationNetwork(args, train_loader, test_loader):
    print('==========start to generate ClassificationNetwork==========')
    net = ClassifyNet(args.num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        net.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.classify_epoch)

    train_total = 0
    train_correct = 0
    test_total = 0
    test_correct = 0
    running_loss = 0.
    best_pre = 0.
    for epoch in range(args.classify_epoch):
        tic = time.time()
        this_lr = optimizer.param_groups[0]['lr']
        this_lr_str = "{:.5e}".format(this_lr)
        net.train()
        for data, targets, index in train_loader:
            # print(data.shape)
            targets = targets.to(torch.float32)
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()

            _, pre_label = net(data)
            _, pre_true_label = torch.max(pre_label, 1)
            train_total += targets.size(0)
            _, true_targets = torch.max(targets, 1)
            train_correct += (pre_true_label == true_targets).sum().item()

            loss = criterion(pre_label, targets)
            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()
        train_pre = train_correct / train_total
        scheduler.step()

        if epoch % args.test_map == args.test_map - 1:
            training_time = time.time() - tic
            tic = time.time()
            net.eval()
            with torch.no_grad():
                for data, targets, index in test_loader:
                    data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                    _, pre_true_label = torch.max(net(data)[1], 1)
                    _, true_targets = torch.max(targets, 1)
                    test_total += targets.size(0)
                    test_correct += (pre_true_label == true_targets).sum().item()
                test_pre = test_correct / test_total
            if test_pre > best_pre:
                best_pre = test_pre
            testing_time = time.time() - tic
            logger.info('[iter:{}/{}][dataset:{}][lr:{}][loss:{:.2f}][train_pre:{:.4f}%][test_pre:{:.4f}%][best_pre:{:.4f}%][training_time:{:.2f}][testing_time:{:.2f}]'.format(
                epoch + 1,
                args.classify_epoch,
                args.dataset,
                this_lr_str,
                running_loss / args.test_map,
                100 * train_pre,
                100 * test_pre,
                100 * best_pre,
                training_time,
                testing_time,
            ))
            running_loss = 0.
    os.makedirs(f'./save/ClassificationNet/', exist_ok=True)
    torch.save(net, f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt')
    print('==========success generate ClassificationNetwork==========')
    return net


def _normalize_similarity_rows(S, num_classes):
    """Row-wise z-score-like normalization; forces the diagonal to 1."""
    S = S.clone()
    mask = torch.eye(num_classes, dtype=torch.bool, device=S.device)
    S[mask] = 0
    for i in range(num_classes):
        S_max = S[i].max()
        S_min = S[i].min()
        S_mean = S[i].mean()
        denom = max(abs((S_max - S_mean).item()), abs((S_min - S_mean).item()), 1e-8)
        S[i] = (S[i] - S_mean) / denom
    S[mask] = 1
    return S


def _build_classifier_similarity(args, net, train_loader):
    """Single pass over train_loader: masked-softmax data-dependent S_cls plus per-class confidence."""
    S = torch.zeros(args.num_classes, args.num_classes).to(args.device)
    conf_sum = torch.zeros(args.num_classes).to(args.device)
    conf_count = torch.zeros(args.num_classes).to(args.device)
    net.eval()
    with torch.no_grad():
        for data, targets, index in train_loader:
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            batch_size = targets.shape[0]
            p_dis, probs = net(data)  # p_dis是logits，probs是softmax(p_dis)
            _, true_targets = torch.max(targets, 1)
            for i in range(batch_size):
                c = true_targets[i]
                tmp = p_dis[i].clone()
                tmp[c] = float('-inf')  # 对最大值做mask（变成-INF）
                S[c] += F.softmax(tmp, dim=0)
                conf_sum[c] += probs[i, c]
                conf_count[c] += 1

    S = (S + S.T) / 2
    S = _normalize_similarity_rows(S, args.num_classes)
    class_confidence = conf_sum / conf_count.clamp(min=1)
    return S, class_confidence


def _get_class_names(args):
    dataset_name = args.dataset.lower()
    if 'cifar' in dataset_name:
        from data.data_loader import _resolve_cifar100_root, _unpickle
        root = _resolve_cifar100_root(args.root)
        meta = _unpickle(os.path.join(root, 'meta'))
        names = meta.get('fine_label_names')
        if names is None:
            raise FileNotFoundError('CIFAR-100 meta file does not contain "fine_label_names".')
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in names]
        names = [n.replace('_', ' ') for n in names]
        return names
    logger.warning(
        'No class-name mapping is available for dataset "{}"; falling back to placeholder names '
        '("class 0", "class 1", ...) for the CLIP text prompts. The text prior will carry little '
        'semantic signal until this dataset\'s class names are wired up.'.format(args.dataset)
    )
    return [f'class {i}' for i in range(args.num_classes)]


def _load_clip_model(args):
    try:
        import clip
    except ImportError as e:
        raise ImportError(
            'The "clip" package is required for --use-adaptive-fusion. Install it with '
            '`pip install git+https://github.com/openai/CLIP.git`, or precompute a vision-language '
            'similarity cache elsewhere and pass it via --vl-similarity-path.'
        ) from e
    clip_model_name = getattr(args, 'clip_model', 'ViT-B/32')
    model, _ = clip.load(clip_model_name, device=args.device)
    model = model.float()
    model.eval()
    return model, clip_model_name


def _to_clip_input(images, device):
    mean = _IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = _IMAGENET_STD.view(1, 3, 1, 1).to(device)
    images = images * std + mean  # undo ImageNet normalize -> back to [0, 1]
    images = images.clamp(0, 1)

    clip_mean = _CLIP_MEAN.view(1, 3, 1, 1).to(device)
    clip_std = _CLIP_STD.view(1, 3, 1, 1).to(device)
    images = (images - clip_mean) / clip_std
    return images


def _build_clip_image_prototypes(args, clip_model, train_loader):
    input_res = clip_model.visual.input_resolution
    sums = torch.zeros(args.num_classes, clip_model.visual.output_dim, device=args.device)
    counts = torch.zeros(args.num_classes, device=args.device)
    with torch.no_grad():
        for data, targets, index in train_loader:
            data, targets = data.to(args.device), targets.to(args.device)
            clip_input = _to_clip_input(data, args.device)
            if clip_input.shape[-1] != input_res or clip_input.shape[-2] != input_res:
                clip_input = F.interpolate(clip_input, size=(input_res, input_res), mode='bicubic', align_corners=False)
            feats = clip_model.encode_image(clip_input).float()
            feats = F.normalize(feats, dim=1)
            _, true_targets = torch.max(targets, 1)
            sums.index_add_(0, true_targets, feats)
            counts.index_add_(0, true_targets, torch.ones_like(true_targets, dtype=torch.float32))
    counts = counts.clamp(min=1).unsqueeze(1)
    prototypes = sums / counts
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes


def _build_clip_text_prototypes(args, clip_model, class_names):
    import clip
    prompts = [_CLIP_PROMPT_TEMPLATE.format(name) for name in class_names]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(args.device)
        feats = clip_model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=1)
    return feats


def _compute_adaptive_weights(args, class_confidence):
    reliability = class_confidence.clamp(0, 1)
    vl_weight = args.vl_fusion_max_weight * (1 - reliability)
    cls_weight = (1 - vl_weight).clamp(min=args.vl_fusion_min_cls_weight, max=1.0)
    vl_weight = 1 - cls_weight

    image_ratio = getattr(args, 'vl_image_text_ratio', 0.5)
    image_weight = vl_weight * image_ratio
    text_weight = vl_weight * (1 - image_ratio)
    return cls_weight, image_weight, text_weight


def _fuse_similarity_matrices(args, S_cls, S_img, S_text, class_confidence):
    cls_weight, image_weight, text_weight = _compute_adaptive_weights(args, class_confidence)
    logger.info('per-class classifier weight (min/mean/max): {:.3f}/{:.3f}/{:.3f}'.format(
        cls_weight.min().item(), cls_weight.mean().item(), cls_weight.max().item()))

    def _pairwise(w):
        return (w.view(-1, 1) + w.view(1, -1)) / 2

    alpha = _pairwise(cls_weight)
    beta = _pairwise(image_weight)
    gamma = _pairwise(text_weight)

    S = alpha * S_cls + beta * S_img.to(S_cls.device) + gamma * S_text.to(S_cls.device)
    mask = torch.eye(args.num_classes, dtype=torch.bool, device=S.device)
    S[mask] = 1
    return S


def GenerateSimilarityMatrix(args, train_loader, test_loader):
    if os.path.exists(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt'):
        print('==========ClassificationNet has already generated==========')
        net = torch.load(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt').to(args.device)
    else:
        net = TrainClassificationNetwork(args, train_loader, test_loader)

    print('==========start to generate SimilarityMatrix==========')
    S_cls, class_confidence = _build_classifier_similarity(args, net, train_loader)

    if not getattr(args, 'use_adaptive_fusion', False):
        S = S_cls
    else:
        vl_cache_path = args.vl_similarity_path or f'./save/SimilarityMatrix/{args.dataset}_VL_Similarity_Cache.pt'
        if os.path.exists(vl_cache_path) and not args.force_rebuild_similarity:
            print(f'==========loading cached vision-language similarity from {vl_cache_path}==========')
            vl_cache = torch.load(vl_cache_path)
            S_img = vl_cache['S_image'].to(args.device)
            S_text = vl_cache['S_text'].to(args.device)
        else:
            clip_model, clip_model_name = _load_clip_model(args)
            class_names = _get_class_names(args)
            print(f'==========building CLIP ({clip_model_name}) vision-language priors==========')
            image_prototypes = _build_clip_image_prototypes(args, clip_model, train_loader)
            text_prototypes = _build_clip_text_prototypes(args, clip_model, class_names)
            S_img = _normalize_similarity_rows(image_prototypes @ image_prototypes.T, args.num_classes)
            S_text = _normalize_similarity_rows(text_prototypes @ text_prototypes.T, args.num_classes)

            os.makedirs(os.path.dirname(vl_cache_path) or '.', exist_ok=True)
            torch.save({'S_image': S_img.cpu(), 'S_text': S_text.cpu(), 'clip_model': clip_model_name}, vl_cache_path)

        S = _fuse_similarity_matrices(args, S_cls, S_img, S_text, class_confidence)

    torch.save(S, get_similarity_matrix_path(args))
    print('==========success generate SimilarityMatrix==========')
    return S
