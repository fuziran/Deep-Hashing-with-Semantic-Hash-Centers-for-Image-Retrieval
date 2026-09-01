import numpy as np
import torch
import torch.nn as nn


def compute_result(dataloader, net, device):
    """
    Generate binary hash codes and labels for a dataloader.
    Return:
      binary_codes: torch.Tensor, shape [N, bit], values in {-1, +1}
      labels:       torch.Tensor, shape [N, num_classes], one-hot
    """
    net.eval()

    binary_list = []
    label_list = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)

            outputs = net(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            binary = torch.sign(outputs.cpu())
            binary[binary == 0] = 1

            binary_list.append(binary)
            label_list.append(labels.cpu())

    return torch.cat(binary_list, dim=0), torch.cat(label_list, dim=0)


def _calc_ap_per_query(query_binary, query_label, retrieval_binary, retrieval_label, topk):
    num_query = query_label.shape[0]
    num_retrieval = retrieval_label.shape[0]
    bit = query_binary.shape[1]

    if topk == -1 or topk is None or topk > num_retrieval:
        topk = num_retrieval

    per_query = np.full(num_query, np.nan, dtype=np.float64)

    retrieval_binary_t = retrieval_binary.T

    for i in range(num_query):
        # relevant images: share at least one label
        gnd = (np.dot(retrieval_label, query_label[i]) > 0).astype(np.float32)

        if gnd.sum() == 0:
            continue

        hamm = 0.5 * (bit - np.dot(query_binary[i], retrieval_binary_t))
        ind = np.argsort(hamm, kind="stable")[:topk]
        gnd = gnd[ind]

        relevant_positions = np.where(gnd == 1)[0] + 1
        if len(relevant_positions) == 0:
            per_query[i] = 0.0
            continue

        precision_at_relevant = np.arange(1, len(relevant_positions) + 1) / relevant_positions
        per_query[i] = precision_at_relevant.mean()

    return per_query


def _calc_map_for_one_topk(query_binary, query_label, retrieval_binary, retrieval_label, topk):
    per_query = _calc_ap_per_query(
        query_binary, query_label, retrieval_binary, retrieval_label, topk
    )
    valid = np.isfinite(per_query)

    if not valid.any():
        return 0.0
    return float(per_query[valid].mean())


def _calc_pr_curve(query_binary, query_label, retrieval_binary, retrieval_label, num_database):
    """
    Compute a lightweight Precision-Recall curve for topK values used in hashing papers.
    """
    num_query = query_label.shape[0]
    bit = query_binary.shape[1]
    retrieval_binary_t = retrieval_binary.T

    max_k = min(500, retrieval_binary.shape[0])
    topk_list = [1, 5, 10] + list(range(20, max_k + 1, 10))
    topk_list = sorted(set([k for k in topk_list if k <= max_k]))

    precision_sum = np.zeros(len(topk_list), dtype=np.float64)
    recall_sum = np.zeros(len(topk_list), dtype=np.float64)
    valid_query = 0

    for i in range(num_query):
        gnd_all = (np.dot(retrieval_label, query_label[i]) > 0).astype(np.float32)
        total_relevant = gnd_all.sum()

        if total_relevant == 0:
            continue

        hamm = 0.5 * (bit - np.dot(query_binary[i], retrieval_binary_t))
        ind = np.argsort(hamm, kind="stable")[:max_k]
        gnd_sorted = gnd_all[ind]
        cumsum = np.cumsum(gnd_sorted)

        for j, k in enumerate(topk_list):
            hit = cumsum[k - 1]
            precision_sum[j] += hit / k
            recall_sum[j] += hit / total_relevant

        valid_query += 1

    if valid_query == 0:
        precision = precision_sum
        recall = recall_sum
    else:
        precision = precision_sum / valid_query
        recall = recall_sum / valid_query

    return {
        "topK": np.array(topk_list),
        "precision": precision,
        "recall": recall,
    }


def CalcTopMapWithPR(query_binary, query_label, retrieval_binary, retrieval_label, topK, num_database=None):
    """
    Match train.py call:
      mAP_list, PR_data = CalcTopMapWithPR(
          tst_binary, tst_label,
          trn_binary, trn_label,
          args.topK, num_database
      )

    topK default in run.py is [-1, 100, 1000].
    Return:
      mAP_list = [mAP@ALL, mAP@100, mAP@1000]
      PR_data
    """
    query_binary = np.asarray(query_binary).astype(np.float32)
    retrieval_binary = np.asarray(retrieval_binary).astype(np.float32)
    query_label = np.asarray(query_label).astype(np.float32)
    retrieval_label = np.asarray(retrieval_label).astype(np.float32)

    if isinstance(topK, int):
        topk_list = [topK]
    else:
        topk_list = list(topK)

    map_list = []
    for k in topk_list:
        map_list.append(
            _calc_map_for_one_topk(
                query_binary,
                query_label,
                retrieval_binary,
                retrieval_label,
                k,
            )
        )

    pr_data = _calc_pr_curve(
        query_binary,
        query_label,
        retrieval_binary,
        retrieval_label,
        num_database,
    )

    return map_list, pr_data


def CalcTopMap(retrieval_binary, query_binary, retrieval_label, query_label, topK):
    """
    Compatibility function for the commented call in train.py.
    """
    if isinstance(topK, int):
        topk_list = [topK]
    else:
        topk_list = list(topK)

    return [
        _calc_map_for_one_topk(
            query_binary,
            query_label,
            retrieval_binary,
            retrieval_label,
            k,
        )
        for k in topk_list
    ]


def CalcTopMapPerQuery(query_binary, query_label, retrieval_binary, retrieval_label, topK):
    """Return one AP vector per requested cutoff for audit and paired statistics."""
    if isinstance(topK, int):
        topk_list = [topK]
    else:
        topk_list = list(topK)
    return {
        str(k): _calc_ap_per_query(
            np.asarray(query_binary, dtype=np.float32),
            np.asarray(query_label, dtype=np.float32),
            np.asarray(retrieval_binary, dtype=np.float32),
            np.asarray(retrieval_label, dtype=np.float32),
            k,
        )
        for k in topk_list
    }
