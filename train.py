import copy

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.linalg import hadamard
import random
from data.data_loader import load_data

torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Innovation 2: Bit-balanced, bit-independent CSQ loss
# =============================================================================

class CSQLoss(torch.nn.Module):
    """Center-based hash loss with bit-balance and bit-independence regularizers.

    Loss = center_loss
         + λ        * Q_loss          (quantization,  original)
         + α_bal    * balance_loss     (Innovation 2: each bit ≈ zero-mean)
         + α_ind    * independence_loss (Innovation 2: bits decorrelated)
    """

    def __init__(self, args, bit, hash_center):
        super(CSQLoss, self).__init__()
        self.is_single_label = True
        self.hash_targets = hash_center.to(args.device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.criterion = torch.nn.BCELoss().to(args.device)

    def forward(self, u, y, ind, args):
        u_tanh = u.tanh()
        hash_center = self.label2center(y)

        # Original: center alignment + quantization
        center_loss = self.criterion(0.5 * (u_tanh + 1), 0.5 * (hash_center + 1))
        Q_loss = (u_tanh.abs() - 1).pow(2).mean()

        loss = center_loss + args.lambd * Q_loss

        # ── Innovation 2a: Bit-balance loss ───────────────────────────────
        # Each bit should be zero-centred across the batch (equal +1 / -1 prob)
        alpha_bal = getattr(args, 'alpha_bal', 0.0)
        if alpha_bal > 0:
            bit_mean = u_tanh.mean(dim=0)           # [bit]
            balance_loss = bit_mean.pow(2).mean()
            loss = loss + alpha_bal * balance_loss

        # ── Innovation 2b: Bit-independence loss ──────────────────────────
        # Bits should be decorrelated; correlation matrix ≈ identity
        alpha_ind = getattr(args, 'alpha_ind', 0.0)
        if alpha_ind > 0:
            u_center = u_tanh - u_tanh.mean(dim=0, keepdim=True)
            cov = (u_center.T @ u_center) / u_tanh.size(0)     # [bit, bit]
            diag = torch.diag(cov).clamp(min=1e-8)
            corr = cov / (diag.unsqueeze(0) * diag.unsqueeze(1)).sqrt()
            eye = torch.eye(corr.size(0), device=corr.device)
            independence_loss = (corr - eye).pow(2).mean()
            loss = loss + alpha_ind * independence_loss

        return loss

    def label2center(self, y):
        return self.hash_targets[y.argmax(axis=1)]

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    break
        return hash_targets


# =============================================================================
# Innovation 3: NT-Xent contrastive consistency loss
# =============================================================================

def _contrastive_loss(u1, u2, tau):
    """NT-Xent loss between two hash-code batches from the same images.

    Positive pairs: (u1[i], u2[i]).  Negatives: all other pairs in the batch.
    Forces hash codes to be invariant to different augmentations.
    """
    u1_norm = F.normalize(u1.tanh(), dim=1)   # [B, bit]
    u2_norm = F.normalize(u2.tanh(), dim=1)   # [B, bit]
    sim = u1_norm @ u2_norm.T / tau            # [B, B]
    labels = torch.arange(u1.size(0), device=u1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
    return loss


# =============================================================================
# Innovation 4: Online adaptive hash center refinement (EMA)
# =============================================================================

class AdaptiveHashCenter:
    """Tracks per-class hash-code centroids and periodically refines the
    target hash centers via Exponential Moving Average (EMA).

    The motivation: hash centers are generated offline from the classification
    feature space (stage 2). As the hashing network trains, its feature
    manifold gradually diverges from the classifier's manifold, widening the
    gap between the fixed centers and the actual learned distribution.
    EMA refinement closes this gap without discarding the distance-optimal
    layout produced by the ADMM solver.
    """

    def __init__(self, hash_center, num_classes, momentum=0.995):
        self.centers     = hash_center.clone().float().cpu()   # [n_class, bit]
        self.momentum    = momentum
        self.num_classes = num_classes
        self.bit         = hash_center.size(1)
        self._reset_accum()

    def _reset_accum(self):
        self.class_sums   = torch.zeros(self.num_classes, self.bit)
        self.class_counts = torch.zeros(self.num_classes)

    def update(self, hash_codes, labels):
        """Accumulate batch statistics (called every training step)."""
        with torch.no_grad():
            codes  = hash_codes.tanh().detach().cpu()     # [B, bit]
            c_ids  = labels.argmax(dim=1).cpu()           # [B]
            for i, c in enumerate(c_ids):
                self.class_sums[c]   += codes[i]
                self.class_counts[c] += 1

    def refine(self):
        """EMA-update centers from accumulated epoch statistics.

        Returns the new hash-center tensor (binarised ±1).
        """
        with torch.no_grad():
            valid = self.class_counts > 0
            if valid.sum() == 0:
                return self.centers

            epoch_mean = self.class_sums[valid] / self.class_counts[valid].unsqueeze(1)
            self.centers[valid] = (
                self.momentum * self.centers[valid]
                + (1.0 - self.momentum) * epoch_mean
            )
            # Binarise: keep centers in {-1, +1}
            new_centers = torch.sign(self.centers)
            new_centers[new_centers == 0] = 1.0
            self.centers = new_centers.clone()

        self._reset_accum()
        return self.centers


# =============================================================================
# Main training function
# =============================================================================

def train_val(args, hash_center, train_loader, test_loader, database_loader, num_database):
    print('==========start to train SHC NetWork==========')
    bit = args.code_length
    net = args.net(bit).to(args.device)

    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)

    hash_center = hash_center.to(torch.float32)
    if hash_center.shape != (args.num_classes, args.code_length):
        hash_center = hash_center.t()   # n_class × bit

    criterion = CSQLoss(args, bit, hash_center)

    # ── Innovation 3: detect dual-view training ───────────────────────────
    alpha_cont     = getattr(args, 'alpha_cont', 0.0)
    tau            = getattr(args, 'tau', 0.07)
    use_contrastive = alpha_cont > 0.0

    # ── Innovation 4: online center refinement setup ───────────────────────
    refine_interval  = getattr(args, 'refine_interval', 0)
    refine_momentum  = getattr(args, 'refine_momentum', 0.995)
    use_center_refine = refine_interval > 0
    if use_center_refine:
        adaptive_center = AdaptiveHashCenter(
            hash_center, args.num_classes, momentum=refine_momentum
        )
        warmup_epochs = args.epoch // 4   # refine only after 25% of training

    # ── Book-keeping ──────────────────────────────────────────────────────
    result_dic    = {}
    Best_mAP_ALL  = 0
    Best_mAP_100  = 0
    Best_mAP_1000 = 0
    mAP_ALL_list  = []
    mAP_100_list  = []
    mAP_1000_list = []
    lr_values     = []
    loss_list     = []

    for epoch in range(0, args.epoch):
        this_lr     = optimizer.param_groups[0]['lr']
        lr_values.append(this_lr)
        this_lr_str = "{:.5e}".format(this_lr)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, Lr:%s, training...." % (
            args.info, epoch + 1, args.epoch, current_time, bit, args.dataset, this_lr_str), end="")

        net.train()
        train_loss = 0

        for batch_data in train_loader:
            optimizer.zero_grad()

            # ── Unpack batch (single-view or dual-view) ───────────────────
            if use_contrastive and len(batch_data) == 4:
                image, image2, label, ind = batch_data
                image  = image.to(args.device)
                image2 = image2.to(args.device)
                label  = label.to(args.device)

                u1 = net(image)
                u2 = net(image2)

                loss = criterion(u1, label.float(), ind, args)
                loss = loss + alpha_cont * _contrastive_loss(u1, u2, tau)
                u_main = u1
            else:
                if len(batch_data) == 4:
                    image, _, label, ind = batch_data   # ignore second view
                else:
                    image, label, ind = batch_data
                image = image.to(args.device)
                label = label.to(args.device)

                u_main = net(image)
                loss   = criterion(u_main, label.float(), ind, args)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Accumulate statistics for Innovation 4
            if use_center_refine:
                adaptive_center.update(u_main, label)

        scheduler.step()
        train_loss = train_loss / len(train_loader)
        loss_list.append(train_loss)
        print("\b\b\b\b\b\b\b loss:%.5f" % train_loss)

        # ── Innovation 4: periodic EMA center refinement ──────────────────
        if use_center_refine and (epoch + 1) % refine_interval == 0 and epoch >= warmup_epochs:
            new_centers = adaptive_center.refine().to(args.device)
            criterion.hash_targets = new_centers
            print(f"  [Innov-4] Hash centers refined at epoch {epoch + 1}.")

        # ── Evaluation ────────────────────────────────────────────────────
        if (epoch + 1) % args.test_map == 0:
            net.eval()
            with torch.no_grad():
                tst_binary, tst_label = compute_result(test_loader,     net, device=args.device)
                trn_binary, trn_label = compute_result(database_loader, net, device=args.device)
                mAP_list, PR_data = CalcTopMapWithPR(
                    tst_binary.numpy(), tst_label.numpy(),
                    trn_binary.numpy(), trn_label.numpy(),
                    args.topK, num_database,
                )
                mAP_ALL  = mAP_list[0]
                mAP_100  = mAP_list[1]
                mAP_1000 = mAP_list[2]
                mAP_ALL_list.append(mAP_ALL)
                mAP_100_list.append(mAP_100)
                mAP_1000_list.append(mAP_1000)

            if mAP_ALL > Best_mAP_ALL:
                Best_mAP_ALL = mAP_ALL
                best_net = copy.deepcopy(net)
            if mAP_100  > Best_mAP_100:
                Best_mAP_100  = mAP_100
            if mAP_1000 > Best_mAP_1000:
                Best_mAP_1000 = mAP_1000

            print(f"{args.info} epoch:{epoch + 1} bit:{bit} dataset:{args.dataset}")
            print(f"MAP ALL:{mAP_ALL:.5f}  Best MAP ALL: {Best_mAP_ALL:.5f}")
            print(f"MAP 100:{mAP_100:.5f}  Best MAP 100: {Best_mAP_100:.5f}")
            print(f"MAP 1000:{mAP_1000:.5f} Best MAP 1000: {Best_mAP_1000:.5f}")

        if (epoch + 1) % args.epoch == 0:
            print('[SHC] final_mAP_ALL:%.5f'  % Best_mAP_ALL)
            print('[SHC] final_mAP_100:%.5f'  % Best_mAP_100)
            print('[SHC] final_mAP_1000:%.5f' % Best_mAP_1000)

    # ── Final evaluation with best model ──────────────────────────────────
    tst_binary, tst_label = compute_result(test_loader,     best_net, device=args.device)
    trn_binary, trn_label = compute_result(database_loader, best_net, device=args.device)
    _, best_PR_data = CalcTopMapWithPR(
        tst_binary.numpy(), tst_label.numpy(),
        trn_binary.numpy(), trn_label.numpy(),
        args.topK, num_database,
    )

    result_dic['loss']     = loss_list
    result_dic['mAP@all']  = mAP_ALL_list
    result_dic['mAP@100']  = mAP_100_list
    result_dic['mAP@1000'] = mAP_1000_list
    result_dic['PR_data']  = best_PR_data
    result_dic['net']      = best_net

    os.makedirs(f'./save/result_log/{args.dataset}', exist_ok=True)
    torch.save(result_dic, f'./save/result_log/{args.dataset}/SHC_bit_{bit}_result_dic.pt')

    # ── Plots ─────────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(list(range(args.epoch)), lr_values)
    plt.xlabel('epoch'); plt.ylabel('lr'); plt.title('LR schedule')

    plt.figure()
    plt.plot(list(range(args.epoch)), loss_list, label='train loss')
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Loss curve')
    plt.show()

    print('==========finish to train SHC NetWork==========')
