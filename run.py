import argparse
import os.path

import torch
from network import *
from data.data_loader import load_data
from GenerateSimilarityMatrix import GenerateSimilarityMatrix
from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from train import train_val


def load_config():
    parser = argparse.ArgumentParser(description='SHC_PyTorch')

    # ── Basic ──────────────────────────────────────────────────────────────
    parser.add_argument('--seed', default=60, type=int)
    parser.add_argument('--info', default='[SHC]', type=str)
    parser.add_argument('--dataset', default='cifar-100-new-seg', type=str,
                        help='Dataset name')
    parser.add_argument('--num-classes', default=100, type=int,
                        help='Number of dataset categories')
    parser.add_argument('--root', default='../data/cifar/cifar-100/cifar-100-new-seg/', type=str,
                        help='Path of dataset')
    parser.add_argument('--code-length', default=32, type=int,
                        help='Binary hash code length')
    parser.add_argument('--lr', default=7e-5, type=float,
                        help='Learning rate (7e-5 for CIFAR-100, 1e-4 for others)')
    parser.add_argument('--epoch', default=300, type=int,
                        help='Training epochs of the hashing network (stage 3)')
    parser.add_argument('--classify-epoch', default=300, type=int,
                        help='Training epochs of the classification network (stage 1)')
    parser.add_argument('--test-map', default=5, type=int,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--beta', default=1.00, type=float)
    parser.add_argument('--lambd', default=0.0001, type=float,
                        help='Quantization loss weight')
    parser.add_argument('--topK', default=[-1, 100, 1000], type=int, nargs='+',
                        help='Calculate mAP at top-K')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--resize-size', default=256, type=int)
    parser.add_argument('--crop-size', default=224, type=int)
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id (-1 for CPU)')

    # ── Innovation 2: Bit balance & independence ───────────────────────────
    parser.add_argument('--alpha-bal', default=0.01, type=float,
                        help='[Innov-2] Bit-balance loss weight (0 = disabled)')
    parser.add_argument('--alpha-ind', default=0.001, type=float,
                        help='[Innov-2] Bit-independence loss weight (0 = disabled)')

    # ── Innovation 3: Contrastive hash consistency ────────────────────────
    parser.add_argument('--alpha-cont', default=0.1, type=float,
                        help='[Innov-3] Contrastive consistency loss weight (0 = disabled)')
    parser.add_argument('--tau', default=0.07, type=float,
                        help='[Innov-3] Temperature for NT-Xent contrastive loss')

    # ── Innovation 4: Online adaptive hash center refinement ──────────────
    parser.add_argument('--refine-interval', default=20, type=int,
                        help='[Innov-4] Epochs between hash-center EMA refinements (0 = disabled)')
    parser.add_argument('--refine-momentum', default=0.995, type=float,
                        help='[Innov-4] EMA momentum for center refinement')

    # ── Innovation 5: Semantic diffusion similarity matrix ────────────────
    parser.add_argument('--use-diffusion', default=True, type=lambda x: x.lower() != 'false',
                        help='[Innov-5] Enable high-order semantic diffusion (default: True)')
    parser.add_argument('--diff-alpha', default=0.15, type=float,
                        help='[Innov-5] Diffusion decay factor')
    parser.add_argument('--diff-steps', default=3, type=int,
                        help='[Innov-5] Number of diffusion hops')
    parser.add_argument('--sim-topk', default=0, type=int,
                        help='[Innov-5] Top-K neighbours per class (0 = auto: max(10, n//10))')

    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.gpu is None or args.gpu < 0:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # ── Auto sim_topk ─────────────────────────────────────────────────────
    if args.sim_topk == 0:
        args.sim_topk = max(10, args.num_classes // 10)

    # ── classify_epoch alias (argparse stores classify-epoch -> classify_epoch) ──
    args.classify_epoch = args.classify_epoch

    # ── Network ───────────────────────────────────────────────────────────
    args.net = ResNet

    return args


if __name__ == '__main__':
    args = load_config()

    # Load data
    train_loader, test_loader, database_loader, num_train, num_test, num_database = load_data(args)

    # Stage 1: Construct the Data-dependent Pairwise Similarity Matrix
    # Cache filename encodes diffusion params so any change triggers regeneration.
    if args.use_diffusion:
        _sim_suffix = f'_diff_a{args.diff_alpha}_s{args.diff_steps}_k{args.sim_topk}'
    else:
        _sim_suffix = '_nodiff'
    _sim_path = f'./save/SimilarityMatrix/{args.dataset}_Similarity_Matrix{_sim_suffix}.pt'

    if os.path.exists(_sim_path):
        print('==========SimilarityMatrix has already generated==========')
        S = torch.load(_sim_path)
    else:
        S = GenerateSimilarityMatrix(args, train_loader, test_loader)
        os.makedirs('./save/SimilarityMatrix/', exist_ok=True)
        torch.save(S, _sim_path)
    S = S.to(args.device)

    # Stage 2: Generate the Semantic Hash Centers
    if os.path.exists(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt'):
        print('==========SHC HashCenters has already generated==========')
        H = torch.load(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt')
    else:
        H = GenerateSemanticHashCenters(args, S)
    H = H.to(args.device)

    # Stage 3: Train the Deep Hashing Network
    train_val(args, H, train_loader, test_loader, database_loader, num_database)
