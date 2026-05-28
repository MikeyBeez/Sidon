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


def l_sidon_k3(embedding_table, vocab_size=None, address_dim=4, num_samples=10000, gamma=1.0):
    """Sidon k=3 regularizer: push apart triple-multiset sums by at least gamma.

    Uses sampling — exhaustive enumeration of pairs of triple-multisets is too large.
    """
    addr = embedding_table[:, :address_dim]
    V = addr.shape[0] if vocab_size is None else vocab_size
    addr = addr[:V]

    n = num_samples
    a = torch.stack([
        torch.randint(V, (n,), device=addr.device),
        torch.randint(V, (n,), device=addr.device),
        torch.randint(V, (n,), device=addr.device),
    ], dim=1).sort(dim=1).values
    b = torch.stack([
        torch.randint(V, (n,), device=addr.device),
        torch.randint(V, (n,), device=addr.device),
        torch.randint(V, (n,), device=addr.device),
    ], dim=1).sort(dim=1).values

    different = (a != b).any(dim=1)
    a = a[different]
    b = b[different]

    if a.shape[0] == 0:
        return torch.tensor(0.0, device=addr.device, requires_grad=True)

    sum_a = addr[a[:, 0]] + addr[a[:, 1]] + addr[a[:, 2]]
    sum_b = addr[b[:, 0]] + addr[b[:, 1]] + addr[b[:, 2]]

    dists = torch.norm(sum_a - sum_b, dim=1)
    violations = torch.clamp(gamma - dists, min=0)
    return violations.mean()


def make_pos_mod(order_dim, n_pos=3):
    """Fixed, distinct per-position modulation vectors (unit-norm rows).
    Deterministic across train/eval: generated on CPU from a fixed seed."""
    g = torch.Generator().manual_seed(12345)
    m = torch.randn(n_pos, order_dim, generator=g)
    m = m / m.norm(dim=1, keepdim=True)
    return m


def l_order_k3(embedding_table, vocab_size=None, sidon_dim=4, order_dim=4,
               num_samples=10000, gamma=1.0):
    """Order-injectivity loss: push apart the position-modulated order pools of
    two orderings of the SAME multiset. Operates on embedding dims [sidon_dim:sidon_dim+order_dim].

    Different multisets are separated by the Sidon channel; this loss only needs to
    separate orderings WITHIN a multiset, which is where the multiplicative position
    modulation breaks the permutation-invariance of the additive pool.
    """
    V = embedding_table.shape[0] if vocab_size is None else vocab_size
    o = embedding_table[:V, sidon_dim:sidon_dim + order_dim]
    pos_mod = make_pos_mod(order_dim).to(embedding_table.device).type_as(o)

    n = num_samples
    t = torch.randint(V, (n, 3), device=embedding_table.device)
    perm = torch.argsort(torch.rand(n, 3, device=embedding_table.device), dim=1)
    t2 = torch.gather(t, 1, perm)
    different = (t != t2).any(dim=1)  # ordering actually changed
    t = t[different]
    t2 = t2[different]
    if t.shape[0] == 0:
        return torch.tensor(0.0, device=embedding_table.device, requires_grad=True)

    pool1 = pos_mod[0] * o[t[:, 0]] + pos_mod[1] * o[t[:, 1]] + pos_mod[2] * o[t[:, 2]]
    pool2 = pos_mod[0] * o[t2[:, 0]] + pos_mod[1] * o[t2[:, 1]] + pos_mod[2] * o[t2[:, 2]]
    dist = torch.norm(pool1 - pool2, dim=1)
    return torch.clamp(gamma - dist, min=0).mean()


@torch.no_grad()
def order_metrics_k3(embedding_table, vocab_size=None, sidon_dim=4, order_dim=4,
                     gamma=1.0, noise_std=0.01, n_eval=3000, block=256,
                     rung1_shared=False):
    """Ordered recovery from the pooled vector alone.

    The recovery procedure receives ONLY the noised pooled vector and a table of
    candidate pools built from the model's parameters. No token/position/order-code
    metadata is supplied.

    rung1_shared=True: no separate order dims; the position modulation is applied to
    the shared Sidon dims (tests whether order falls out of multiset geometry for free).
    rung1_shared=False (rung 2): Sidon dims carry the multiset (plain sum), separate
    order dims carry the position-modulated order pool; the full vector is their concat.
    """
    V = embedding_table.shape[0] if vocab_size is None else vocab_size
    addr = embedding_table[:V, :sidon_dim]
    device = embedding_table.device

    # Enumerate all V^3 ordered triples
    ar = torch.arange(V, device=device)
    ii, jj, kk = torch.meshgrid(ar, ar, ar, indexing='ij')
    triples = torch.stack([ii.flatten(), jj.flatten(), kk.flatten()], dim=1)  # (V^3, 3)
    N = triples.shape[0]

    if rung1_shared:
        pos_mod = make_pos_mod(sidon_dim).to(device).type_as(addr)
        full = (pos_mod[0] * addr[triples[:, 0]]
                + pos_mod[1] * addr[triples[:, 1]]
                + pos_mod[2] * addr[triples[:, 2]])
    else:
        o = embedding_table[:V, sidon_dim:sidon_dim + order_dim]
        pos_mod = make_pos_mod(order_dim).to(device).type_as(o)
        ms = addr[triples[:, 0]] + addr[triples[:, 1]] + addr[triples[:, 2]]
        op = (pos_mod[0] * o[triples[:, 0]]
              + pos_mod[1] * o[triples[:, 1]]
              + pos_mod[2] * o[triples[:, 2]])
        full = torch.cat([ms, op], dim=1)

    g = torch.Generator(device=device).manual_seed(0)
    qidx = torch.randperm(N, device=device, generator=g)[:min(n_eval, N)]
    qfull = full[qidx]
    noisy = qfull + torch.randn_like(qfull) * noise_std
    q_sorted = triples[qidx].sort(dim=1).values

    ordered_correct = 0
    multiset_correct = 0
    n_q = qidx.shape[0]
    for s in range(0, n_q, block):
        e = min(s + block, n_q)
        d = torch.cdist(noisy[s:e], full)
        pred = d.argmin(dim=1)
        ordered_correct += (pred == qidx[s:e]).sum().item()
        pred_sorted = triples[pred].sort(dim=1).values
        multiset_correct += (pred_sorted == q_sorted[s:e]).all(dim=1).sum().item()

    return {
        'ordered_recovery': ordered_correct / n_q,
        'multiset_recovery_from_ordered_table': multiset_correct / n_q,
        'n_ordered_triples': N,
        'n_eval': n_q,
        'order_dim': 0 if rung1_shared else order_dim,
    }


def _enumerate_triples(V, device):
    """All multisets of size 3 from V items: (i, j, k) with i <= j <= k.
    Returns tensor of shape (C(V+2, 3), 3)."""
    triples = []
    for i in range(V):
        for j in range(i, V):
            for k in range(j, V):
                triples.append((i, j, k))
    return torch.tensor(triples, dtype=torch.long, device=device)


@torch.no_grad()
def sidon_metrics_k3(embedding_table, vocab_size=None, address_dim=4, gamma=1.0,
                     noise_std=0.01, block_size=512, n_eval_query=5000):
    """Compute Sidon satisfaction rate and triple-multiset recovery accuracy.

    Uses block-wise cdist to fit on 16GB GPU at V=65 (n_multisets ≈ 47k).
    """
    addr = embedding_table[:, :address_dim]
    V = addr.shape[0] if vocab_size is None else vocab_size
    addr = addr[:V]

    triples = _enumerate_triples(V, addr.device)
    sums = addr[triples[:, 0]] + addr[triples[:, 1]] + addr[triples[:, 2]]
    n_sums = sums.shape[0]

    # Block-wise pairwise distance: collect satisfaction, min, mean
    violations_count = 0
    total_pairs = 0
    min_dist = float('inf')
    sum_dist = 0.0
    for start in range(0, n_sums, block_size):
        end = min(start + block_size, n_sums)
        block = sums[start:end]
        dists = torch.cdist(block, sums)  # (block, n_sums)
        # Mask diagonal
        for r in range(end - start):
            dists[r, start + r] = float('inf')
        flat = dists.flatten()
        violations_count += (flat < gamma).sum().item()
        total_pairs += flat.numel()
        block_min = flat.min().item()
        if block_min < min_dist:
            min_dist = block_min
        # Mean: sum non-inf entries
        finite_mask = torch.isfinite(flat)
        sum_dist += flat[finite_mask].sum().item()

    satisfaction_rate = 1 - violations_count / total_pairs
    mean_dist = sum_dist / total_pairs

    # Clean recovery: argmin of distances to all (excluding self) should be self
    # Equivalently: min off-diagonal distance > 0 (sums are unique)
    # Sample subset of queries for memory tractability
    n_eval = min(n_eval_query, n_sums)
    g = torch.Generator(device=addr.device).manual_seed(0)
    query_idx = torch.randperm(n_sums, device=addr.device, generator=g)[:n_eval]

    # Noisy recovery
    query_sums = sums[query_idx]
    noise = torch.randn_like(query_sums) * noise_std
    noisy = query_sums + noise

    correct_noisy = 0
    for start in range(0, n_eval, block_size):
        end = min(start + block_size, n_eval)
        block_q = noisy[start:end]
        dists = torch.cdist(block_q, sums)
        predicted = dists.argmin(dim=1)
        correct_noisy += (predicted == query_idx[start:end]).sum().item()
    noisy_acc = correct_noisy / n_eval

    # Clean recovery: each sum's nearest off-diagonal is at distance > 0
    correct_clean = 0
    for start in range(0, n_eval, block_size):
        end = min(start + block_size, n_eval)
        block_q = sums[query_idx[start:end]]
        dists = torch.cdist(block_q, sums)
        for r in range(end - start):
            dists[r, query_idx[start + r]] = float('inf')
        min_d = dists.min(dim=1).values
        correct_clean += (min_d > 0).sum().item()
    clean_acc = correct_clean / n_eval

    return {
        'sidon_satisfaction_rate': satisfaction_rate,
        'pair_recovery_accuracy_clean': clean_acc,
        f'pair_recovery_accuracy_noisy_{noise_std}': noisy_acc,
        'ordered_recovery_accuracy_noisy': noisy_acc,
        'min_pairwise_distance': min_dist,
        'mean_pairwise_distance': mean_dist,
        'n_multisets': n_sums,
    }


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
