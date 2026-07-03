import argparse
import os.path

import torch
from network import *
from data.data_loader import load_data
from GenerateSimilarityMatrix import GenerateSimilarityMatrix, get_similarity_matrix_path
from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from train import train_val


def load_config():
    parser = argparse.ArgumentParser(description='SHC_PyTorch')
    parser.add_argument('--seed', default=60, type=int,
                        help='seed')
    parser.add_argument('--info', default='[SHC]', type=str,
                        help='information')
    parser.add_argument('--dataset', default='cifar-100-new-seg', type=str,
                        help='Dataset name.(default: cifar-100-new-seg, stanford_cars-new-seg, stanford_cars-official-seg, NAbirds-new-seg, NAbirds-official-seg,')
    parser.add_argument('--num-classes', default=100, type=int,
                        help='num classes of dataset.(default: 100)')
    parser.add_argument('--root', default='../data/cifar/cifar-100/cifar-100-new-seg/', type=str,
                        help='Path of dataset')
    parser.add_argument('--code-length', default=32, type=int,
                        help='Binary hash code length.(default: 32)')
    parser.add_argument('--lr', default=7e-5, type=float,
                        help='Learning rate.(default: 7e-5 for CIFAR-100, 1e-4 for other datasets)')
    parser.add_argument('--epoch', default=300, type=int,
                        help='max epoch.(default: 300)')
    parser.add_argument('--classify_epoch', default=300, type=int,
                        help='max epoch for classification.(default: 300)')
    parser.add_argument('--test-map', default=5, type=int,
                        help='test frequency.(default: 10)')
    parser.add_argument('--beta', default=1.00, type=float,
                        help='para')
    parser.add_argument('--lambd', default=0.0001, type=float,
                        help='para')
    parser.add_argument('--topK', default=[-1, 100, 1000], type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--resize-size', default=256, type=int,
                        help='picture resize size.(default: 256)')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='picture crop size.(default: 224)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: 0)')
    parser.add_argument('--use-adaptive-fusion', action='store_true',
                        help='Use confidence-adaptive fusion between SHC similarity and a vision-language prior.')
    parser.add_argument('--vl-similarity-path', default=None, type=str,
                        help='Optional path of a precomputed class-level vision-language similarity matrix.')
    parser.add_argument('--vl-fusion-max-weight', default=0.25, type=float,
                        help='Maximum weight assigned to the vision-language prior in adaptive fusion.')
    parser.add_argument('--vl-fusion-min-cls-weight', default=0.70, type=float,
                        help='Minimum weight retained for the original data-dependent SHC similarity matrix.')
    parser.add_argument('--force-rebuild-similarity', action='store_true',
                        help='Regenerate the similarity matrix even when a cached matrix exists.')
    parser.add_argument('--clip-model', default='ViT-B/32', type=str,
                        help='CLIP backbone used to build the vision-language prior.(default: ViT-B/32)')
    parser.add_argument('--vl-image-text-ratio', default=0.5, type=float,
                        help='Fraction of the vision-language weight assigned to CLIP image similarity '
                             '(the remainder goes to CLIP text similarity).(default: 0.5)')
    parser.add_argument('--use-semantic-margin-proxy-loss', action='store_true',
                        help='Enable an opt-in proxy-anchor-style negative hinge term in Stage 3 that pushes '
                             'each sample away from every OTHER class hash center, using a per-class-pair margin '
                             'that is smaller for semantically similar classes and larger for dissimilar classes '
                             '(derived from the Stage-1 similarity matrix S and the Stage-2 get_margin bounds). '
                             'Default: off (baseline CSQLoss behavior is unchanged).')
    parser.add_argument('--proxy-margin-weight', default=0.05, type=float,
                        help='Weight applied to the semantic-margin proxy loss term before adding it to '
                             'center_loss + lambd*Q_loss. Only used when --use-semantic-margin-proxy-loss is set. '
                             '(default: 0.05)')
    parser.add_argument('--proxy-margin-similarity-clip', default=1.0, type=float,
                        help='Clip bound applied to S before mapping it to a [0,1] dissimilarity used to '
                             'interpolate between the theoretical d_min/d_max Hamming margins. Only used when '
                             '--use-semantic-margin-proxy-loss is set. (default: 1.0)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # net
    args.net = ResNet

    return args


if __name__ == '__main__':
    args = load_config()
    # load data
    train_loader, test_loader, database_loader, num_train, num_test, num_database = load_data(args)

    # Stage1：Construct the Data-dependent Pairwise Similarity Matrix
    similarity_matrix_path = get_similarity_matrix_path(args)
    if os.path.exists(similarity_matrix_path) and not args.force_rebuild_similarity:
        print('==========SimilarityMatrix has already generated==========')
        S = torch.load(similarity_matrix_path)
    else:
        S = GenerateSimilarityMatrix(args, train_loader, test_loader)
    S = S.to(args.device)

    # Stage 2: Generate the Semantic Hash Centers
    if os.path.exists(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt'):
        print('==========SHC HashCenters has already generated==========')
        H = torch.load(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt')
    else:
        H = GenerateSemanticHashCenters(args, S)
    H = H.to(args.device)

    # Stage 3: Train the Deep Hashing Network
    train_val(args, H, S, train_loader, test_loader, database_loader, num_database)
