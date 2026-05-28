"""Evaluate a trained checkpoint: val perplexity, Sidon metrics, pair recovery."""

import os
import sys
import json
import math
import pickle
import argparse

import numpy as np
import torch

from model import GPTConfig, GPT
from sidon import sidon_metrics, sidon_metrics_k3


def load_checkpoint(ckpt_path, device='cuda'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_args, checkpoint


def compute_val_perplexity(model, data_dir, block_size, batch_size, device, eval_iters=200):
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
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss), mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='shakespeare_char')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--address_dim', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--baseline_perplexity', type=float, default=None)
    parser.add_argument('--sidon_k', type=int, default=2)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    model, model_args, checkpoint = load_checkpoint(ckpt_path, args.device)
    block_size = model_args['block_size']
    vocab_size = model_args['vocab_size']

    data_dir = os.path.join('data', args.dataset)

    print(f"Evaluating {ckpt_path} (vocab={vocab_size}, block_size={block_size})")

    val_ppl, val_loss = compute_val_perplexity(
        model, data_dir, block_size, args.batch_size, args.device, args.eval_iters)
    print(f"Val perplexity: {val_ppl:.4f} (loss: {val_loss:.4f})")

    wte = model.transformer.wte.weight
    if args.sidon_k == 3:
        sm = sidon_metrics_k3(wte, vocab_size=vocab_size, address_dim=args.address_dim,
                              gamma=args.gamma, noise_std=args.noise_std)
        print(f"n_multisets: {sm['n_multisets']}")
        print(f"Sidon satisfaction rate: {sm['sidon_satisfaction_rate']:.4f}")
        print(f"Multiset recovery (clean): {sm['pair_recovery_accuracy_clean']:.4f}")
        print(f"Multiset recovery (noisy {args.noise_std}): {sm[f'pair_recovery_accuracy_noisy_{args.noise_std}']:.4f}")
        print(f"Ordered recovery (noisy {args.noise_std}): {sm['ordered_recovery_accuracy_noisy']:.4f}")
        print(f"Min pairwise distance: {sm['min_pairwise_distance']:.6f}")
    else:
        sm = sidon_metrics(wte, vocab_size=vocab_size, address_dim=args.address_dim,
                           gamma=args.gamma, noise_std=args.noise_std)
        print(f"Sidon satisfaction rate: {sm['sidon_satisfaction_rate']:.4f}")
        print(f"Pair recovery (clean): {sm['pair_recovery_accuracy_clean']:.4f}")
        print(f"Pair recovery (noisy {args.noise_std}): {sm[f'pair_recovery_accuracy_noisy_{args.noise_std}']:.4f}")
        print(f"Min pairwise distance: {sm['min_pairwise_distance']:.6f}")

    log_path = os.path.join(args.ckpt_dir, 'training_log.json')
    loss_curve, lm_loss_curve, sidon_curve = [], [], []
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)
        loss_curve = [e['loss'] for e in training_log]
        lm_loss_curve = [e.get('lm_loss', e['loss']) for e in training_log]
        sidon_curve = [e['sidon_loss'] for e in training_log]

    run_config = checkpoint.get('config', {})
    result = {
        'config': {
            'lambda': run_config.get('sidon_lambda', 0.0),
            'address_dim': args.address_dim,
            'gamma': args.gamma,
            'vocab': args.dataset.replace('shakespeare_', ''),
            'seed': run_config.get('seed', 1337),
            'steps': run_config.get('max_iters', 5000),
        },
        'training': {
            'final_train_loss': loss_curve[-1] if loss_curve else None,
            'final_lm_loss': lm_loss_curve[-1] if lm_loss_curve else None,
            'final_sidon_loss': sidon_curve[-1] if sidon_curve else None,
            'loss_curve': loss_curve,
            'lm_loss_curve': lm_loss_curve,
            'sidon_curve': sidon_curve,
        },
        'evaluation': {
            'val_perplexity': val_ppl,
            'val_loss': val_loss,
            'val_perplexity_baseline_ratio': (val_ppl / args.baseline_perplexity
                                              if args.baseline_perplexity else None),
            **sm,
        }
    }

    output_path = args.output or os.path.join(args.ckpt_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_path}")

    return result


if __name__ == '__main__':
    main()
