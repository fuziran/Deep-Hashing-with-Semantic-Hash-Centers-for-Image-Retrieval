import os.path
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from utils.tools import *
from network import *
from loguru import logger


# ---------------------------------------------------------------------------
# Innovation 5: High-order semantic diffusion helpers
# ---------------------------------------------------------------------------

def _high_order_diffusion(S, alpha=0.15, steps=3):
    """Graph diffusion to propagate transitive semantic relations.

    S_diffused = sum_{k=0}^{steps} alpha^k * T^k
    where T is the row-normalised off-diagonal similarity (random-walk matrix).

    Args:
        S     : [n, n] symmetric similarity matrix, diagonal = 1
        alpha : propagation decay factor (0 < alpha < 1)
        steps : number of diffusion hops

    Returns:
        S_final: [n, n] fused matrix (original weighted 0.7 + diffused 0.3)
    """
    device = S.device
    n = S.size(0)

    S_off = S.clone()
    S_off.fill_diagonal_(0.0)

    row_sum = S_off.sum(dim=1, keepdim=True).clamp(min=1e-8)
    T = S_off / row_sum          # row-normalised transition matrix

    S_diffused = torch.eye(n, device=device)
    T_power = T.clone()
    coeff = alpha
    for _ in range(steps):
        S_diffused = S_diffused + coeff * T_power
        T_power = T_power @ T
        coeff = coeff * alpha

    # symmetrise and restore self-similarity
    S_diffused = (S_diffused + S_diffused.T) / 2.0
    S_diffused.fill_diagonal_(1.0)

    # weighted fusion: keep 70% original signal, add 30% diffused context
    S_final = 0.7 * S + 0.3 * S_diffused
    S_final.fill_diagonal_(1.0)
    return S_final


def _sparse_topk_similarity(S, topk):
    """Keep only top-k neighbours per class; zero-out weak similarities.

    Reduces noise from weakly-related classes while preserving strong
    semantic cues.  The result is re-symmetrised so the matrix stays
    symmetric.
    """
    n = S.size(0)
    S_sparse = torch.zeros_like(S)
    for i in range(n):
        row = S[i].clone()
        row[i] = float('-inf')               # exclude self
        vals, idx = torch.topk(row, k=min(topk, n - 1))
        # only keep neighbours with positive similarity
        mask = vals > 0
        S_sparse[i, idx[mask]] = vals[mask]

    S_sparse = (S_sparse + S_sparse.T) / 2.0
    S_sparse.fill_diagonal_(1.0)
    return S_sparse


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

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
            logger.info(
                '[iter:{}/{}][dataset:{}][lr:{}][loss:{:.2f}]'
                '[train_pre:{:.4f}%][test_pre:{:.4f}%][best_pre:{:.4f}%]'
                '[training_time:{:.2f}][testing_time:{:.2f}]'.format(
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
    os.makedirs('./save/ClassificationNet/', exist_ok=True)
    torch.save(net, f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt')
    print('==========success generate ClassificationNetwork==========')
    return net


def GenerateSimilarityMatrix(args, train_loader, test_loader):
    if os.path.exists(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt'):
        print('==========ClassificationNet has already generated==========')
        net = torch.load(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt').to(args.device)
    else:
        net = TrainClassificationNetwork(args, train_loader, test_loader)

    print('==========start to generate SimilarityMatrix==========')
    S = torch.zeros(args.num_classes, args.num_classes).to(args.device)
    net.eval()
    with torch.no_grad():
        for data, targets, index in train_loader:
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            batch_size = targets.shape[0]
            p_dis, _ = net(data)
            _, true_targets = torch.max(targets, 1)
            for i in range(batch_size):
                tmp = p_dis[i].clone()
                tmp[true_targets[i]] = float('-inf')
                S[true_targets[i]] += F.softmax(tmp, dim=0)

    mask = torch.eye(args.num_classes).bool().to(args.device)
    S = (S + S.T) / 2
    for i in range(args.num_classes):
        S_max = S[i].max()
        S_min = S[i].min()
        S_mean = S[i].mean()
        S[i] = (S[i] - S_mean) / max(abs(S_max - S_mean), abs(S_min - S_mean))
    S[mask] = 1

    # ------------------------------------------------------------------
    # Innovation 5: high-order semantic diffusion + top-k sparsification
    # ------------------------------------------------------------------
    use_diffusion = getattr(args, 'use_diffusion', True)
    if use_diffusion:
        diff_alpha = getattr(args, 'diff_alpha', 0.15)
        diff_steps = getattr(args, 'diff_steps', 3)
        topk_neighbours = getattr(args, 'sim_topk', max(10, args.num_classes // 10))

        print(f'[Innovation 5] Applying semantic diffusion: alpha={diff_alpha}, steps={diff_steps}')
        S = _high_order_diffusion(S, alpha=diff_alpha, steps=diff_steps)

        print(f'[Innovation 5] Applying top-k sparsification: k={topk_neighbours}')
        S = _sparse_topk_similarity(S, topk=topk_neighbours)

        print(f'[Innovation 5] S stats after diffusion — '
              f'min={S[~mask].min():.4f}, max={S[~mask].max():.4f}, '
              f'mean={S[~mask].mean():.4f}')

    print('==========success generate SimilarityMatrix==========')
    return S
