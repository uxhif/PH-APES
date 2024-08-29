import torch


def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_knn(pcd, center, center_idx, k, pts):
    device = pcd.device

    B, C, N = pcd.shape
    idx = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, C, -1)
    mask = torch.stack([~torch.isin(idx[b], center_idx[b]) for b in range(B)], dim=0)
    pcd_without_center = pcd[mask].reshape(B, C, N - center_idx.size(1))

    pcd_without_center = pcd_without_center.permute(0, 2, 1)
    center = center.permute(0, 2, 1).to(device)
    center_idx = center_idx.to(device)

    neighbor_idx = knn(center, pcd_without_center, k)
    neighbor_idx = neighbor_idx.reshape(B, -1)
    M = pts - center_idx.size(1)
    sampled_idx = torch.zeros((B, M), dtype=neighbor_idx.dtype, device=device)

    for b in range(B):
        nbd = k
        neighbor_idx_b = torch.unique(neighbor_idx[b])
        while neighbor_idx_b.size(0) < M:
            # if nbd < pcd_without_center.size(1)/4:
            #     nbd = nbd*2
            # else:
            #     nbd += 4
            nbd += 4
            new_idx = knn(center[b].unsqueeze(0), pcd_without_center[b].unsqueeze(0), nbd).reshape(1, -1)
            new_idx = torch.unique(new_idx)
            neighbor_idx_b = new_idx.squeeze(dim=0)
        unique_idx = torch.unique(neighbor_idx_b)
        sampled_idx[b] = unique_idx[torch.randperm(unique_idx.size(0))[:M]].to(device)   

    sampled_idx = torch.cat((center_idx, sampled_idx), dim=-1)
    knn_pcd = torch.gather(pcd, 2, sampled_idx.unsqueeze(1).expand(-1, 128, -1))

    return knn_pcd


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()
