"""k=3 ordered recovery sweep on char-level Shakespeare."""

import os
import sys
import json
import subprocess
import time

LAMBDAS = [0.0, 0.01, 0.1, 1.0]
SEEDS = [1337, 1338, 1339]
DATASET = 'shakespeare_char'
RESULTS_DIR = os.path.join('..', 'experiments', 'k3_ordered_recovery', 'runs')
SIDON_K = 3

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
    'log_interval': 100,
    'warmup_iters': 500,
    'lr_decay_iters': 20000,
    'min_lr': 3e-5,
    'compile': False,
    'dropout': 0.0,
    'bias': False,
    'sidon_k': SIDON_K,
    'address_dim': 4,
    'sidon_gamma': 1.0,
    'sidon_num_samples': 10000,
    'always_save_checkpoint': True,
}


def run_name(lam, seed):
    return f"k3_char_lambda{lam}_seed{seed}"


def train_one(lam, seed):
    name = run_name(lam, seed)
    out_dir = os.path.join('out', name)
    args = dict(COMMON_ARGS)
    args['sidon_lambda'] = lam
    args['seed'] = seed
    args['out_dir'] = out_dir

    cmd = [sys.executable, 'train.py']
    for k, v in args.items():
        cmd.append(f'--{k}={v}')

    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"Finished {name} in {elapsed:.0f}s (exit code {result.returncode})")
    return result.returncode == 0


def eval_one(lam, seed, baseline_ppl=None):
    name = run_name(lam, seed)
    out_dir = os.path.join('out', name)
    output_path = os.path.join(RESULTS_DIR, f'{name}.json')

    cmd = [sys.executable, 'eval.py',
           f'--ckpt_dir={out_dir}',
           f'--dataset={DATASET}',
           f'--address_dim={COMMON_ARGS["address_dim"]}',
           f'--gamma={COMMON_ARGS["sidon_gamma"]}',
           f'--sidon_k={SIDON_K}',
           f'--output={output_path}']
    if baseline_ppl is not None:
        cmd.append(f'--baseline_perplexity={baseline_ppl}')

    print(f"Evaluating: {name}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Eval failed for {name}")
        return None

    with open(output_path) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failed = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            ok = train_one(lam, seed)
            if not ok:
                failed.append(run_name(lam, seed))

    if failed:
        print(f"\nWARNING: {len(failed)} runs failed: {failed}")

    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    baseline_ppls = []
    for seed in SEEDS:
        result = eval_one(0.0, seed)
        if result:
            baseline_ppls.append(result['evaluation']['val_perplexity'])
    baseline_ppl = sum(baseline_ppls) / len(baseline_ppls) if baseline_ppls else None
    print(f"\nBaseline perplexity (mean over seeds): {baseline_ppl:.4f}")

    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        for seed in SEEDS:
            eval_one(lam, seed, baseline_ppl=baseline_ppl)

    for seed in SEEDS:
        eval_one(0.0, seed, baseline_ppl=baseline_ppl)

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print(f"Results in: {os.path.abspath(RESULTS_DIR)}")
    print("="*60)


if __name__ == '__main__':
    main()
