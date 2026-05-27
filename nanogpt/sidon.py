import torch


def l_sidon(embedding_table, vocab_size=None, address_dim=2, num_samples=10000,
            gamma=1.0, exhaustive_threshold=5000):
    """Sidon k=2 regularizer: push all pairwise multiset sums apart by at least gamma."""
    addr = embedding_table[:, :address_dim]
    V = addr.shape[0] if vocab_size is None else vocab_size
    addr = addr[:V]

    n_sums = V * (V + 1) // 2

    if n_sums <= exhaustive_threshold:
        return _l_sidon_exhaustive(addr, V, gamma)
    else:
        return _l_sidon_sampled(addr, V, num_samples, gamma)


def _l_sidon_exhaustive(addr, V, gamma):
    idx_i, idx_j = torch.triu_indices(V, V, device=addr.device)
    sums = addr[idx_i] + addr[idx_j]
    dists = torch.cdist(sums, sums)
    mask = ~torch.eye(sums.shape[0], dtype=torch.bool, device=addr.device)
    violations = torch.clamp(gamma - dists[mask], min=0)
    return violations.mean()


def _l_sidon_sampled(addr, V, num_samples, gamma):
    idx_a1 = torch.randint(V, (num_samples,), device=addr.device)
    idx_a2 = torch.randint(V, (num_samples,), device=addr.device)
    idx_b1 = torch.randint(V, (num_samples,), device=addr.device)
    idx_b2 = torch.randint(V, (num_samples,), device=addr.device)

    pair_a_lo = torch.min(idx_a1, idx_a2)
    pair_a_hi = torch.max(idx_a1, idx_a2)
    pair_b_lo = torch.min(idx_b1, idx_b2)
    pair_b_hi = torch.max(idx_b1, idx_b2)

    different = (pair_a_lo != pair_b_lo) | (pair_a_hi != pair_b_hi)
    pair_a_lo = pair_a_lo[different]
    pair_a_hi = pair_a_hi[different]
    pair_b_lo = pair_b_lo[different]
    pair_b_hi = pair_b_hi[different]

    if pair_a_lo.shape[0] == 0:
        return torch.tensor(0.0, device=addr.device, requires_grad=True)

    sum_a = addr[pair_a_lo] + addr[pair_a_hi]
    sum_b = addr[pair_b_lo] + addr[pair_b_hi]

    dists = torch.norm(sum_a - sum_b, dim=1)
    violations = torch.clamp(gamma - dists, min=0)
    return violations.mean()


@torch.no_grad()
def sidon_metrics(embedding_table, vocab_size=None, address_dim=2, gamma=1.0,
                  noise_std=0.01):
    """Compute Sidon satisfaction rate and pair recovery accuracy."""
    addr = embedding_table[:, :address_dim]
    V = addr.shape[0] if vocab_size is None else vocab_size
    addr = addr[:V]

    idx_i, idx_j = torch.triu_indices(V, V, device=addr.device)
    sums = addr[idx_i] + addr[idx_j]
    n_sums = sums.shape[0]

    dists = torch.cdist(sums, sums)
    mask = ~torch.eye(n_sums, dtype=torch.bool, device=addr.device)
    satisfaction_rate = (dists[mask] >= gamma).float().mean().item()

    noise = torch.randn_like(sums) * noise_std
    noisy_sums = sums + noise
    nn_dists = torch.cdist(noisy_sums, sums)
    predicted = nn_dists.argmin(dim=1)
    noisy_acc = (predicted == torch.arange(n_sums, device=addr.device)).float().mean().item()

    clean_dists = dists.clone()
    clean_dists[torch.arange(n_sums, device=addr.device), torch.arange(n_sums, device=addr.device)] = float('inf')
    min_clean_dist = clean_dists.min(dim=1).values
    clean_acc = (min_clean_dist > 0).float().mean().item()

    return {
        'sidon_satisfaction_rate': satisfaction_rate,
        'pair_recovery_accuracy_clean': clean_acc,
        f'pair_recovery_accuracy_noisy_{noise_std}': noisy_acc,
        'min_pairwise_distance': dists[mask].min().item(),
        'mean_pairwise_distance': dists[mask].mean().item(),
    }
