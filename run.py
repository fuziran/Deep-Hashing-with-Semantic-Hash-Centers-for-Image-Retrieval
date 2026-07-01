import argparse
import os.path

import torch
from network import *
from data.data_loader import load_data
from GenerateSimilarityMatrix import GenerateSimilarityMatrix
from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from train import train_val

""" 项目主入口
解析命令行参数、设置设备、指定默认网络为ResNet,然后按三阶段顺序执行 """

""" 定义训练参数 """
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
    parser.add_argument('--use-diffusion', default=True, type=lambda x: str(x).lower() != 'false',
                        help='Enable high-order semantic diffusion. Use false to disable.')
    parser.add_argument('--diff-alpha', default=0.15, type=float,
                        help='Semantic diffusion decay factor.')
    parser.add_argument('--diff-steps', default=3, type=int,
                        help='Number of semantic diffusion hops.')
    parser.add_argument('--sim-topk', default=0, type=int,
                        help='Top-k semantic neighbours per class. 0 means max(10, num_classes//10).')
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

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    if args.sim_topk == 0:
        args.sim_topk = max(10, args.num_classes // 10)

    # net
    args.net = ResNet

    return args


if __name__ == '__main__':
    args = load_config()
    """ 加载数据 """
    # load data
    train_loader, test_loader, database_loader, num_train, num_test, num_database = load_data(args)
    """ 如果本地已有 ./save/SimilarityMatrix/...pt，直接读取相似度矩阵；否则调用 GenerateSimilarityMatrix() 生成。 """
    # Stage1：Construct the Data-dependent Pairwise Similarity Matrix
    if args.use_diffusion:
        sim_suffix = f'_diff_a{args.diff_alpha}_s{args.diff_steps}_k{args.sim_topk}'
    else:
        sim_suffix = '_nodiff'
    sim_path = f'./save/SimilarityMatrix/{args.dataset}_Similarity_Matrix{sim_suffix}.pt'
    hash_center_path = f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}{sim_suffix}.pt'
    args.shc_hash_center_path = hash_center_path

    if os.path.exists(sim_path):
        print('==========SimilarityMatrix has already generated==========')
        S = torch.load(sim_path)
    else:
        S = GenerateSimilarityMatrix(args, train_loader, test_loader)
        os.makedirs('./save/SimilarityMatrix/', exist_ok=True)
        torch.save(S, sim_path)
    S = S.to(args.device)

    """ 如果本地已有 ./save/HashCenters/...pt，直接读取哈希中心；否则调用 GenerateSemanticHashCenters() 生成。 """
    # Stage 2: Generate the Semantic Hash Centers
    if os.path.exists(hash_center_path):
        print('==========SHC HashCenters has already generated==========')
        H = torch.load(hash_center_path)
    else:
        H = GenerateSemanticHashCenters(args, S)
        os.makedirs('./save/HashCenters/', exist_ok=True)
        torch.save(H, hash_center_path)
    H = H.to(args.device)

    """ 训练哈希网络 """
    # Stage 3: Train the Deep Hashing Network
    train_val(args, H, train_loader, test_loader, database_loader, num_database)
