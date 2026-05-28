"""k=3 learned-order rung 2: dedicated order subspace + injectivity loss.
Sweeps order_dim in {2, 4, 8} x lambda {0, 0.01, 0.1, 1.0} x 3 seeds.
"""

import os
import sys
import json
import subprocess
import time

LAMBDAS = [0.0, 0.01, 0.1, 1.0]
SEEDS = [1337, 1338, 1339]
ORDER_DIMS = [4, 2, 8]  # 4 first: headline cell, then fill the dimension-cost curve
DATASET = 'shakespeare_char'
RESULTS_DIR = os.path.join('..', 'experiments', 'k3_learned_order', 'runs')
SIDON_DIM = 4

COMMON_ARGS = {
    'dataset': DATASET,
    'batch_size': 64,
    'block_size': 256,
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 128,
    'learning_rate': 3e-4,
    'max_iters': 20000,
    'weight_decay': 0.1,
    'gradient_accumulation_steps': 1,
    'eval_interval': 2000,
    'eval_iters': 200,
    'log_interval': 500,
    'warmup_iters': 500,
    'lr_decay_iters': 20000,
    'min_lr': 3e-5,
    'compile': False,
    'dropout': 0.0,
    'bias': False,
    'sidon_k': 3,
    'address_dim': SIDON_DIM,
    'sidon_gamma': 1.0,
    'sidon_num_samples': 10000,
    'order_gamma': 1.0,
    'order_num_samples': 10000,
    'always_save_checkpoint': True,
}


def run_name(odim, lam, seed):
    return f"rung2_dim{odim}_lambda{lam}_seed{seed}"


def train_one(odim, lam, seed):
    name = run_name(odim, lam, seed)
    out_dir = os.path.join('out', name)
    args = dict(COMMON_ARGS)
    args['sidon_lambda'] = lam
    args['order_dim'] = odim
    args['seed'] = seed
    args['out_dir'] = out_dir
    cmd = [sys.executable, 'train.py'] + [f'--{k}={v}' for k, v in args.items()]
    print(f"\n{'='*60}\nStarting: {name}\n{'='*60}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=False)
    print(f"Finished {name} in {time.time()-t0:.0f}s (exit {r.returncode})")
    return r.returncode == 0


def eval_one(odim, lam, seed, baseline_ppl=None):
    name = run_name(odim, lam, seed)
    out_dir = os.path.join('out', name)
    output_path = os.path.join(RESULTS_DIR, f'{name}.json')
    cmd = [sys.executable, 'eval_order.py',
           f'--ckpt_dir={out_dir}', f'--dataset={DATASET}',
           f'--sidon_dim={SIDON_DIM}', f'--order_dim={odim}',
           '--rung=2', f'--output={output_path}']
    if baseline_ppl is not None:
        cmd.append(f'--baseline_perplexity={baseline_ppl}')
    print(f"Evaluating: {name}")
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f"Eval failed for {name}")
        return None
    with open(output_path) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Train+eval per order_dim block so each dimension's results land complete.
    for odim in ORDER_DIMS:
        for lam in LAMBDAS:
            for seed in SEEDS:
                train_one(odim, lam, seed)

        print(f"\n{'='*60}\nEVAL: order_dim={odim}\n{'='*60}")
        bppls = []
        for seed in SEEDS:
            r = eval_one(odim, 0.0, seed)
            if r:
                bppls.append(r['evaluation']['val_perplexity'])
        bppl = sum(bppls) / len(bppls) if bppls else None
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            for seed in SEEDS:
                eval_one(odim, lam, seed, baseline_ppl=bppl)
        for seed in SEEDS:
            eval_one(odim, 0.0, seed, baseline_ppl=bppl)
        print(f"order_dim={odim} block complete (baseline PPL {bppl})")

    print("\n" + "="*60 + "\nSWEEP COMPLETE\n" + f"Results in: {os.path.abspath(RESULTS_DIR)}\n" + "="*60)


if __name__ == '__main__':
    main()
