import os.path
import torch
import random
from scipy.linalg import hadamard, eig
import numpy as np
from scipy.special import comb
import copy
import time
from utils.experiment import build_cache_metadata, load_cache, save_cache


# np.random.seed(4)
# random.seed(4)
# torch.manual_seed(4)

def get_margin(bit, n_class):
    # 1. 计算d_max
    L = bit
    right = (2 ** L) / n_class
    # print(f"border is {right}")
    d_min = 0
    d_max = 0
    for j in range(2 * L + 4):
        dim = j
        sum_1 = 0
        sum_2 = 0
        for i in range((dim - 1) // 2 + 1):
            sum_1 += comb(L, i)
        for i in range((dim) // 2 + 1):
            sum_2 += comb(L, i)
        if sum_1 <= right and sum_2 > right:
            d_min = dim
    for i in range(2 * L + 4):
        dim = i
        sum_1 = 0
        sum_2 = 0
        for j in range(dim):
            sum_1 += comb(L, j)
        for j in range(dim - 1):
            sum_2 += comb(L, j)
        if sum_1 >= right and sum_2 < right:
            # `print(f"sum 1 is {sum_1}")
            # print(f"sum 2 is {sum_2}")`
            d_max = dim
            break
    # 2. 计算alpha_neg和alpha_pos
    alpha_neg = L - 2 * d_max
    # beta_neg = L - 2 * d_min
    alpha_pos = L
    # print(d_min)
    return d_max, d_min


def CSQ_init(n_class, bit):
    h_k = hadamard(bit)
    h_2k = np.concatenate((h_k, -h_k), 0)
    hash_center = h_2k[:n_class]

    if h_2k.shape[0] < n_class:
        hash_center = np.resize(hash_center, (n_class, bit))
        for k in range(5):
            for index in range(h_2k.shape[0], n_class):
                ones = np.ones(bit)
                ones[random.sample(list(range(bit)), bit // 2)] = -1
                hash_center[index] = ones
            c = []
            for i in range(n_class):
                for j in range(i + 1, n_class):
                    c.append(sum(hash_center[i] != hash_center[j]))
            c = np.array(c)
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                break
    return hash_center


def init_hash(n_class, bit):
    hash_centers = -1 + 2 * np.random.random((n_class, bit))
    hash_centers = np.sign(hash_centers)
    return hash_centers


# @jit(nopython=True)
def cal_Cx(x, H):
    return np.dot(H, x)


# @jit(nopython=True)
def cal_M(H):
    return np.dot(H.T, H) / H.shape[0]


# @jit(nopython=True)
def cal_b(H):
    """
    求H中所有哈希中心的均值
    """
    return np.dot(np.ones(H.shape[0], dtype=np.float64), H) / H.shape[0]


# @jit(nopython=True)
def cal_one_hamm(b, H):
    temp = 0.5 * (b.shape[0] - np.dot(H, b))
    return temp.mean() + temp.min() - temp.var(), temp.min()


# @jit(nopython=True)
def cal_hamm(H):
    dist = []
    for i in range(H.shape[0]):
        for j in range(i + 1, H.shape[0]):
            TF = np.sum(H[i] != H[j])
            dist.append(TF)
    dist = np.array(dist)
    st = dist.mean() + dist.min() - dist.var()
    # print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}")
    return st, dist.mean(), dist.min(), dist.var(), dist.max()


# @jit(nopython=True)
def in_range(z1, z2, z3, bit):
    """
    截断误差
    """
    flag = True
    for item in z1:
        if item < -1 or item > 1:
            flag = False
            return flag
    for item in z3:
        if item < 0:
            flag = False
            return flag
    res = 0
    for item in z2:
        res += item ** 2
    if abs(res - bit) > 0.001:
        flag = False
        return flag
    return flag


# @jit(nopython=True)
def get_min(b, H):
    temp = []
    for i in range(H.shape[0]):
        TF = np.sum(b != H[i])
        temp.append(TF)
    temp = np.array(temp)
    # print(temp.min())
    return temp.min()


# @jit(nopython=True)
def Lp_box_one(b, H, d_max, n_class, bit, rho, gamma, error):
    """
    要优化的哈希中心为x
    """
    b = b.astype(np.float64)
    H = H.astype(np.float64)
    # 计算下界
    d = bit - 2 * d_max

    # 先优化一个哈希中心
    M = cal_M(H)  # M的维度是 n x n
    C = cal_b(H)  # b的维度是 n x 1
    out_iter = 1000
    in_iter = 10
    upper_rho = 1e7
    learning_fact = 1.09
    count = 0
    best_eval, best_min = cal_one_hamm(np.sign(b), H)
    best_B = b
    # 每一次迭代的初始参数都是一样的
    # 将辅助变量z1，z2初始化为输入哈希中心
    z1 = b.copy()
    z2 = b.copy()
    # 初始化每个输入哈希中心与其他哈希中心的内积，这些内积离下界的差距，维度为 m-1
    z3 = d - cal_Cx(np.sign(b), H)
    # 对拉格朗日乘子进行随机初始化
    y1 = np.random.rand(bit)
    y2 = np.random.rand(bit)
    y3 = np.random.rand(n_class - 1)
    # 转换数据类型
    z1 = z1.astype(np.float64)
    z2 = z2.astype(np.float64)
    z3 = z3.astype(np.float64)
    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)
    y3 = y3.astype(np.float64)

    # 外层迭代
    for e in range(out_iter):
        for ei in range(in_iter):
            # 更新x
            left = ((rho + rho) * np.eye(bit, dtype=np.float64) + rho * np.dot(H.T, H))
            left = left.astype(np.float64)
            right = (rho * z1 + rho * z2 + rho * np.dot(H.T, (d - z3)) - y1 - y2 - np.dot(H.T, y3) - C)
            right = right.astype(np.float64)
            b = np.dot(np.linalg.inv(left), right)

            # 更新z1
            z1 = b + 1 / rho * y1
            # 更新z2
            z2 = b + 1 / rho * y2
            # 更新z3
            z3 = d - np.dot(H, b) - 1 / rho * y3

            if in_range(z1, z2, z3, bit):
                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
                break
            else:
                z1[z1 > 1] = 1
                z1[z1 < -1] = -1

                norm_x = np.linalg.norm(z2)
                z2 = np.sqrt(bit) * z2 / norm_x

                z3[z3 < 0] = 0

                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
        # 更新rho
        rho = min(learning_fact * rho, upper_rho)
        if rho == upper_rho:
            count += 1
            eval, mini = cal_one_hamm(np.sign(b), H)
            if eval > best_eval or mini > best_min:
                best_eval = eval
                best_min = mini
                best_B = np.sign(b)
        if max(np.linalg.norm(b - z1, np.inf),
               max(np.linalg.norm(b - z2, np.inf), np.linalg.norm(np.dot(H, b) + z3 - d, np.inf))) < error:
            break
        if count == 100:
            # best_B = np.sign(b)
            break
    # best_B = np.sign(b)
    return best_B, H


# @jit(nopython=True)
def Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st):
    count = 0
    for oo in range(50):
        for i in range(n_class):
            H = np.vstack((B[:i], B[i + 1:]))  # m-1 x n
            B[i], _ = Lp_box_one(B[i], H, d_max, n_class, bit, rho, gamma, error)
        eval_st, eval_mean, eval_min, eval_var, eval_max = cal_hamm(B)
        print(eval_st, eval_min, eval_mean, eval_var, eval_max)
        if eval_st > best_st:
            best_st = eval_st
            best_B = B.copy()
            count = 0
        else:
            count += 1
        if eval_min >= d_max:
            break
    return best_B

def GetRawHashCenter(bit, n_class):
    if bit & (bit - 1) == 0:
        return CSQ_init(n_class, bit)
    else:
        return init_hash(n_class, bit)


def pair_quadratic_gradient(h_i, h_j):
    """Return 2 * (h_j h_j^T) h_i without collapsing it to ||h_j||² h_i."""
    return 2 * h_j * torch.dot(h_j, h_i)


def pairwise_hamming_distances(centers):
    """Return unique off-diagonal Hamming distances for q x C centers."""
    bit, num_classes = centers.shape
    distance_matrix = (bit - centers.T @ centers) / 2
    mask = torch.triu(
        torch.ones(num_classes, num_classes, dtype=torch.bool, device=centers.device),
        diagonal=1,
    )
    return distance_matrix[mask]


def semantic_similarity_loss(similarity, centers):
    bit = centers.shape[0]
    return ((similarity - centers.T @ centers / bit) ** 2).mean()


def keep_better_centers(best_centers, best_loss, candidate_centers, candidate_loss):
    """Keep the raw/best candidate unless an iteration strictly improves it."""
    candidate_value = float(candidate_loss)
    if candidate_value < float(best_loss):
        return candidate_centers.clone(), candidate_value, True
    return best_centers, float(best_loss), False


def loss_is_not_worse(final_loss, reference_loss, atol=1e-7, rtol=1e-6):
    """Compare float32 losses across CPU/GPU without rejecting roundoff noise."""
    final_value = float(final_loss)
    reference_value = float(reference_loss)
    tolerance = atol + rtol * abs(reference_value)
    return final_value <= reference_value + tolerance


def select_monotonic_discrete_update(
    centers,
    center_index,
    gradient,
    similarity,
    min_distance,
    improvement_tolerance=1e-7,
):
    """Find the best feasible descent candidate induced by one center update.

    A fixed projected-gradient step can leave every bit unchanged when the
    gradient does not cross the sign threshold.  Enumerating all distinct
    prefix projections along the descent direction removes that scale
    dependency.  Single-bit candidates provide a deterministic coordinate
    descent fallback.  The returned candidate is accepted only when it lowers
    the exact semantic objective and preserves the minimum Hamming distance.
    """
    bit, num_classes = centers.shape
    current = centers[:, center_index]
    aligned = current * gradient > 0
    candidate_rows = []

    if torch.any(aligned):
        aligned_indices = torch.where(aligned)[0]
        strengths = gradient[aligned_indices].abs()
        order = torch.argsort(strengths, descending=True, stable=True)
        ordered_indices = aligned_indices[order]
        ordered_strengths = strengths[order]
        projected = current.clone()
        start = 0
        while start < ordered_indices.numel():
            end = start + 1
            while (
                end < ordered_indices.numel()
                and ordered_strengths[end] == ordered_strengths[start]
            ):
                end += 1
            projected[ordered_indices[start:end]] *= -1
            candidate_rows.append(projected.clone())
            start = end

    for bit_index in range(bit):
        candidate = current.clone()
        candidate[bit_index] *= -1
        candidate_rows.append(candidate)

    candidates = torch.unique(torch.stack(candidate_rows), dim=0)
    correlations = candidates @ centers / bit
    correlations[:, center_index] = 1.0
    target_row = similarity[center_index]
    candidate_row_errors = ((target_row[None, :] - correlations) ** 2).sum(dim=1)

    current_correlations = current @ centers / bit
    current_correlations[center_index] = 1.0
    current_row_error = ((target_row - current_correlations) ** 2).sum()

    distances = (bit - candidates @ centers) / 2
    distances[:, center_index] = float("inf")
    feasible = distances.amin(dim=1) >= min_distance
    improving = candidate_row_errors < current_row_error - improvement_tolerance
    eligible = feasible & improving

    diagnostics = {
        "candidate_count": int(candidates.shape[0]),
        "constraint_rejections": int((~feasible).sum().item()),
        "loss_rejections": int((feasible & ~improving).sum().item()),
        "gradient_abs_mean": float(gradient.abs().mean().item()),
        "gradient_abs_max": float(gradient.abs().max().item()),
        "accepted": False,
        "flipped_bits": 0,
        "loss_delta": 0.0,
    }
    if not torch.any(eligible):
        return current.clone(), diagnostics

    eligible_errors = candidate_row_errors.masked_fill(~eligible, float("inf"))
    selected_index = int(torch.argmin(eligible_errors).item())
    selected = candidates[selected_index]
    # Each off-diagonal entry occurs in both the row and column of symmetric S.
    loss_delta = (
        2
        * (candidate_row_errors[selected_index] - current_row_error)
        / (num_classes ** 2)
    )
    diagnostics.update(
        {
            "accepted": True,
            "flipped_bits": int((selected != current).sum().item()),
            "loss_delta": float(loss_delta.item()),
        }
    )
    return selected, diagnostics

def GetMinimalDistanceHashCenter(args):
    metadata = build_cache_metadata(args, "mds")
    cache_path = os.path.join(
        args.output_dir,
        "HashCenters",
        f"{args.dataset}_MDS_{metadata['config_hash']}.pt",
    )
    if not args.force_recompute:
        cached = load_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded audited MDS center cache: {cache_path}")
            return cached
    print('==========start to generate MDS HashCenters==========')
    bit = args.code_length
    n_class = args.num_classes
    initWithCSQ = True
    if bit & (bit - 1) != 0:
        initWithCSQ = False
    d_max, d_min = get_margin(bit, n_class)
    d_max = d_max
    print(f"d_max is {d_max}, d_min is {d_min}")
    # 参数初始化
    rho = 1e-5
    gamma = (1 + 5 ** 0.5) / 2
    error = 1e-6
    # 初始化哈希中心
    random.seed(args.seed)
    np.random.seed(args.seed)
    d = bit - 2 * d_max
    if initWithCSQ:
        B = CSQ_init(n_class, bit)  # initialize with CSQ
    else:
        B = init_hash(n_class, bit)  # random initialization

    # 初始评价指标
    best_st, best_mean, best_min, best_var, best_max = cal_hamm(B)
    best_B = copy.deepcopy(B)
    count = 0
    error_index = {}
    print(
        f"best_st is {best_st}, best_min is {str(best_min)}, best_mean is {best_mean}, best_var is {best_var}, best_max is {str(best_max)}")
    best_st = 0
    print(f"eval st, eval min, eval mean, eval var, eval max")
    begin = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(begin))
    print(time_string)
    best_B = Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st)
    end = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    print(time_string)
    ev_st, ev_mean, ev_min, ev_var, ev_max = cal_hamm(best_B)
    print(
        f"ev_st is {ev_st}, ev_min is {str(ev_min)}, ev_mean is {ev_mean}, ev_var is {ev_var}, ev_max is {str(ev_max)}")
    save_cache(cache_path, best_B, metadata)
    print('==========success generate MDS HashCenters==========')
    return best_B

def GenerateSemanticHashCenters(args, S):

    metadata = build_cache_metadata(args, "centers")
    cache_path = os.path.join(
        args.output_dir,
        "HashCenters",
        f"{args.dataset}_SHC_{metadata['config_hash']}.pt",
    )
    if not args.force_recompute:
        cached = load_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded audited SHC center cache: {cache_path}")
            return cached

    d, _ = get_margin(args.code_length, args.num_classes)
    print('min_d is: ', d)

    S = S.to(args.device)
    H = GetMinimalDistanceHashCenter(args)
    print('==========start to generate SHC HashCenters==========')
    H = torch.from_numpy(H).to(torch.float32)
    if H.shape != (args.code_length, args.num_classes):
        H = H.T
    device = args.device
    H = H.to(device)  # q * c

    K = args.code_length - 2 * d - H.T @ H
    K.fill_diagonal_(0)

    center_update_strategy = getattr(
        args, "center_update_strategy", "monotonic_discrete_search"
    )
    if center_update_strategy not in {
        "paper_projected_gradient",
        "monotonic_discrete_search",
    }:
        raise ValueError(
            f"Unsupported center update strategy: {center_update_strategy}"
        )
    # Algorithm 1 initializes lambda to 0.1; the audited B0 repair uses zero.
    lambda_initial = 0.1 if center_update_strategy == "paper_projected_gradient" else 0.0
    lambd = torch.full(
        (args.code_length, args.num_classes),
        lambda_initial,
        device=device,
        dtype=torch.float32,
    )
    rho = 0.2
    miu = 0.625
    alpha = torch.zeros(
        args.num_classes, args.num_classes, device=device, dtype=torch.float32
    )
    beta = torch.full(
        (args.num_classes, args.num_classes), 1e-6,
        device=device, dtype=torch.float32,
    )
    print('center update strategy: ', center_update_strategy)
    epochs = 20
    inner_epochs = 3
    eta = 0.5

    loss1 = semantic_similarity_loss(S, H).item()
    min_loss = (
        float("inf")
        if center_update_strategy == "paper_projected_gradient"
        else loss1
    )
    best_H = H.clone()
    best_epoch = -1
    center_history = []
    gram = H.T @ H
    loss2 = gram.sum() - gram.diag().sum()
    hamming_distances = pairwise_hamming_distances(H)
    gen_min_d = hamming_distances.min()
    gen_max_d = hamming_distances.max()
    print('raw')
    print('loss1: ', loss1)
    print('loss2: ', loss2.item())
    print('min_d: ', gen_min_d.item())
    print('max_d: ', gen_max_d.item())
    center_history.append(
        {
            "epoch": -1,
            "candidate": "raw_mds",
            "similarity_loss": loss1,
            "min_hamming_distance": gen_min_d.item(),
            "max_hamming_distance": gen_max_d.item(),
            "attempted_updates": 0,
            "accepted_updates": 0,
            "constraint_rejections": 0,
            "loss_rejections": 0,
            "unchanged_updates": 0,
            "candidate_evaluations": 0,
            "flipped_bits": 0,
            "gradient_abs_mean": 0.0,
            "gradient_abs_max": 0.0,
            "is_best": center_update_strategy != "paper_projected_gradient",
        }
    )

    for epoch in range(epochs):
        # M-step
        M = torch.inverse((2 / (args.code_length ** 2) * H @ H.T + rho * torch.eye(args.code_length).to(device))) @ (2 / args.code_length * H @ S + lambd + rho * H)
        M = M.to(device)

        # K-step
        K = torch.max((args.code_length - 2 * d - H.T @ H + alpha / beta), torch.zeros(args.num_classes, args.num_classes).to(device))
        K = K.to(device)

        # H-step
        attempted_updates = 0
        accepted_updates = 0
        constraint_rejections = 0
        loss_rejections = 0
        unchanged_updates = 0
        candidate_evaluations = 0
        flipped_bits = 0
        gradient_abs_mean_sum = 0.0
        gradient_abs_max = 0.0
        for inner_epoch in range(inner_epochs):
            for i in range(args.num_classes):
                # i = 99 - i
                sum = torch.zeros(args.code_length).to(device)
                for j in range(args.num_classes):
                    if j == i:
                        continue
                    else:
                        sum += (
                            2 * miu * H[:, j]
                            - 2 * alpha[i, j] * H[:, j]
                            + beta[i, j]
                            * (
                                pair_quadratic_gradient(H[:, i], H[:, j])
                                - 2
                                * (args.code_length - 2 * d - K[i, j])
                                * H[:, j]
                            )
                        )
                if center_update_strategy == "paper_projected_gradient":
                    # The released implementation scales this pairwise term by C^2.
                    sum /= args.num_classes ** 2
                der = (2 / (args.code_length ** 2) * M @ M.T @ H[:, i] - 2 / args.code_length * M @ S[:, i]) + lambd[:, i] + rho * (H[:, i] - M[:, i]) + sum

                attempted_updates += 1
                if center_update_strategy == "paper_projected_gradient":
                    projected = torch.sign(H[:, i] - der / eta)
                    projected = torch.where(projected == 0, H[:, i], projected)
                    candidate = H.clone()
                    candidate[:, i] = projected
                    violates_distance = (
                        pairwise_hamming_distances(candidate).min().item() < d
                    )
                    changed_bits = int((projected != H[:, i]).sum().item())
                    accepted = not violates_distance and changed_bits > 0
                    update_diagnostics = {
                        "candidate_count": 1,
                        "constraint_rejections": int(violates_distance),
                        "loss_rejections": 0,
                        "gradient_abs_mean": float(der.abs().mean().item()),
                        "gradient_abs_max": float(der.abs().max().item()),
                        "accepted": accepted,
                        "flipped_bits": changed_bits if accepted else 0,
                    }
                else:
                    projected, update_diagnostics = select_monotonic_discrete_update(
                        H, i, der, S, d
                    )
                candidate_evaluations += update_diagnostics["candidate_count"]
                constraint_rejections += update_diagnostics["constraint_rejections"]
                loss_rejections += update_diagnostics["loss_rejections"]
                gradient_abs_mean_sum += update_diagnostics["gradient_abs_mean"]
                gradient_abs_max = max(
                    gradient_abs_max, update_diagnostics["gradient_abs_max"]
                )
                if not update_diagnostics["accepted"]:
                    unchanged_updates += 1
                    continue
                H[:, i] = projected
                accepted_updates += 1
                flipped_bits += update_diagnostics["flipped_bits"]

        # lambd-step
        for i in range(args.num_classes):
            lambd[:, i] = lambd[:, i] + rho * (H[:, i] - M[:, i])

        # alpha-step
        for i in range(args.num_classes):
            for j in range(args.num_classes):
                if i == j:
                    continue
                alpha[i, j] = alpha[i, j] + beta[i, j] * (args.code_length - 2 * d - torch.dot(H[:, i], H[:, j]) - K[i, j])
        loss1 = semantic_similarity_loss(S, H).item()
        gram = H.T @ H
        loss2 = gram.sum() - gram.diag().sum()
        hamming_distances = pairwise_hamming_distances(H)
        gen_min_d = hamming_distances.min()
        gen_max_d = hamming_distances.max()
        print('epoch: ', epoch)
        print('loss1: ', loss1)
        print('loss2: ', loss2.item())
        print('min_d: ', gen_min_d.item())
        print('max_d: ', gen_max_d.item())
        print(
            'updates: ',
            f'attempted={attempted_updates}, accepted={accepted_updates}, '
            f'unchanged={unchanged_updates}, candidates={candidate_evaluations}, '
            f'constraint_rejected={constraint_rejections}, '
            f'loss_rejected={loss_rejections}, flipped_bits={flipped_bits}',
        )
        print(
            'gradient: ',
            f'abs_mean={gradient_abs_mean_sum / max(attempted_updates, 1):.8f}, '
            f'abs_max={gradient_abs_max:.8f}',
        )
        best_H, min_loss, improved = keep_better_centers(
            best_H, min_loss, H, loss1
        )
        if improved:
            best_epoch = epoch
        center_history.append(
            {
                "epoch": epoch,
                "candidate": "semantic_update",
                "similarity_loss": loss1,
                "min_hamming_distance": gen_min_d.item(),
                "max_hamming_distance": gen_max_d.item(),
                "attempted_updates": attempted_updates,
                "accepted_updates": accepted_updates,
                "constraint_rejections": constraint_rejections,
                "loss_rejections": loss_rejections,
                "unchanged_updates": unchanged_updates,
                "candidate_evaluations": candidate_evaluations,
                "flipped_bits": flipped_bits,
                "gradient_abs_mean": gradient_abs_mean_sum
                / max(attempted_updates, 1),
                "gradient_abs_max": gradient_abs_max,
                "is_best": improved,
            }
        )

    H = best_H
    H = H.to('cpu')
    S = S.to('cpu')
    hamming_distances = pairwise_hamming_distances(H)
    gen_min_d = hamming_distances.min()
    gen_max_d = hamming_distances.max()
    gen_var_d = hamming_distances.var()
    similarity_match = semantic_similarity_loss(S, H)
    mean_hamming_distance = hamming_distances.mean()
    min_d_fre = torch.sum(torch.eq(hamming_distances, gen_min_d))
    print('gen_min_d: ', gen_min_d.item())
    print('gen_max_d: ', gen_max_d.item())
    print('gen_var_d: ', gen_var_d.item())
    print('similarity_match: ', similarity_match.item())
    print('mean_hamming_distance: ', mean_hamming_distance.item())
    print('min_d_fre: ', min_d_fre.item())
    if not torch.all((H == -1) | (H == 1)):
        raise RuntimeError("Saved SHC centers are not binary {-1,+1}")
    if gen_min_d.item() < d:
        raise RuntimeError(
            f"Saved SHC centers violate minimum distance: {gen_min_d.item()} < {d}"
        )
    if center_update_strategy == "monotonic_discrete_search" and not loss_is_not_worse(
        similarity_match.item(), center_history[0]["similarity_loss"]
    ):
        raise RuntimeError("Saved SHC centers are worse than the raw MDS centers")
    save_cache(cache_path, H, metadata)
    torch.save(
        {
            "min_hamming_distance": gen_min_d,
            "max_hamming_distance": gen_max_d,
            "hamming_variance": gen_var_d,
            "similarity_loss": similarity_match,
            "mean_hamming_distance": mean_hamming_distance,
            "raw_similarity_loss": center_history[0]["similarity_loss"],
            "best_epoch": best_epoch,
            "history": center_history,
        },
        os.path.join(args.run_dir, "stage2_diagnostics.pt"),
    )
    print('==========success generate SHC HashCenters==========')
    return H
