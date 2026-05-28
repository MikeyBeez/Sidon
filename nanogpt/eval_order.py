"""Ordered-recovery evaluation for the k=3 learned-order ladder.

Rung 1: re-evaluate existing k=3 checkpoints (no order training) with
position-modulated recovery on the shared Sidon dims.
Rung 2: evaluate checkpoints trained with a dedicated order subspace.
"""

import os
import sys
import json
import math
import argparse

import numpy as np
import torch

from model import GPTConfig, GPT
from sidon import order_metrics_k3, sidon_metrics_k3


def load_checkpoint(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    sd = ckpt['model']
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, ckpt['model_args'], ckpt


def compute_val_perplexity(model, data_dir, block_size, device, batch_size=64, eval_iters=200):
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    m = sum(losses) / len(losses)
    return math.exp(m), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--dataset', default='shakespeare_char')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sidon_dim', type=int, default=4)
    ap.add_argument('--order_dim', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--noise_std', type=float, default=0.01)
    ap.add_argument('--rung', type=int, required=True, choices=[1, 2])
    ap.add_argument('--output', default=None)
    ap.add_argument('--baseline_perplexity', type=float, default=None)
    args = ap.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}")
        sys.exit(1)

    model, model_args, ckpt = load_checkpoint(ckpt_path, args.device)
    block_size = model_args['block_size']
    vocab_size = model_args['vocab_size']
    data_dir = os.path.join('data', args.dataset)

    print(f"Evaluating {ckpt_path} (rung {args.rung}, vocab={vocab_size})")
    val_ppl, val_loss = compute_val_perplexity(model, data_dir, block_size, args.device)
    print(f"Val PPL: {val_ppl:.4f}")

    wte = model.transformer.wte.weight

    # multiset reference (Sidon channel, plain sum)
    ms = sidon_metrics_k3(wte, vocab_size=vocab_size, address_dim=args.sidon_dim,
                          gamma=args.gamma, noise_std=args.noise_std)
    multiset_recovery = ms[f'pair_recovery_accuracy_noisy_{args.noise_std}']
    print(f"Multiset recovery (Sidon channel): {multiset_recovery:.4f}")

    rung1 = (args.rung == 1)
    om = order_metrics_k3(wte, vocab_size=vocab_size, sidon_dim=args.sidon_dim,
                          order_dim=args.order_dim, gamma=args.gamma,
                          noise_std=args.noise_std, rung1_shared=rung1)
    print(f"Ordered recovery: {om['ordered_recovery']:.4f}  "
          f"(over {om['n_ordered_triples']} ordered triples, {om['n_eval']} queries)")
    print(f"Multiset recovery (from ordered table): {om['multiset_recovery_from_ordered_table']:.4f}")

    run_cfg = ckpt.get('config', {})
    result = {
        'config': {
            'rung': args.rung,
            'lambda': run_cfg.get('sidon_lambda', 0.0),
            'sidon_dim': args.sidon_dim,
            'order_dim': 0 if rung1 else args.order_dim,
            'gamma': args.gamma,
            'seed': run_cfg.get('seed', 1337),
            'recovery_inputs': 'noised pooled vector + model-derived candidate table; '
                               'NO token/position/order-code metadata',
        },
        'evaluation': {
            'val_perplexity': val_ppl,
            'val_loss': val_loss,
            'val_perplexity_baseline_ratio': (val_ppl / args.baseline_perplexity
                                              if args.baseline_perplexity else None),
            'multiset_recovery_sidon': multiset_recovery,
            'sidon_satisfaction_rate': ms['sidon_satisfaction_rate'],
            'ordered_recovery': om['ordered_recovery'],
            'multiset_recovery_from_ordered_table': om['multiset_recovery_from_ordered_table'],
            'ordered_vs_multiset_gap': multiset_recovery - om['ordered_recovery'],
            'n_ordered_triples': om['n_ordered_triples'],
        },
    }

    out = args.output or os.path.join(args.ckpt_dir, f'order_results_rung{args.rung}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
