"""
CRS vs Model v2.1b Comparison Evaluator:
Computes evasion metrics from outputs/crs_results.csv:
- CRS evasion rate on seeds vs on mutations
- Model evasion rate on seeds vs on mutations
- Hybrid (blocked by CRS OR model) evasion rate
- Counts of: evades CRS only / evades model only / evades both
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRS vs Model v2.1b Comparative Evasion")
    parser.add_argument("--crs_csv", default="outputs/crs_results.csv",
                        help="Path to crs_results.csv produced by replay_crs.py")
    args = parser.parse_args()

    if not os.path.exists(args.crs_csv):
        print(f"[ERROR] Results file not found: {args.crs_csv}")
        sys.exit(1)

    df = pd.read_csv(args.crs_csv)
    print(f"Loaded {len(df):,d} replayed records from {args.crs_csv}.\n")

    # In replay_payloads.csv:
    # A row is a seed if id == seed_id (or if mutated_payload == original_payload)
    is_seed = (df['id'] == df['seed_id']) | (df['original_payload'] == df['mutated_payload'])
    df['is_seed'] = is_seed

    # Columns:
    # crs_blocked = (http_status == 403)
    # crs_evaded = (http_status != 403) and (http_status != 0)
    # model_evaded = True/False

    print("="*105)
    print("CRS vs MODEL v2.1b COMPARATIVE EVASION BENCHMARK")
    print("="*105)

    for attack_type in df['attack_type'].unique():
        sub = df[df['attack_type'] == attack_type].copy()
        seeds = sub[sub['is_seed']]
        muts = sub[~sub['is_seed']]

        n_seeds = len(seeds)
        n_muts = len(muts)

        # 1. Seeds
        crs_evade_seeds = int(seeds['crs_evaded'].sum()) if n_seeds > 0 else 0
        crs_rate_seeds = (crs_evade_seeds / n_seeds) * 100.0 if n_seeds > 0 else 0.0

        model_evade_seeds = int(seeds['model_evaded'].sum()) if n_seeds > 0 else 0
        model_rate_seeds = (model_evade_seeds / n_seeds) * 100.0 if n_seeds > 0 else 0.0

        # Hybrid blocked: blocked by CRS OR blocked by Model
        # A request evades hybrid if it evades BOTH CRS AND Model
        hybrid_evade_seeds = int((seeds['crs_evaded'] & seeds['model_evaded']).sum()) if n_seeds > 0 else 0
        hybrid_rate_seeds = (hybrid_evade_seeds / n_seeds) * 100.0 if n_seeds > 0 else 0.0

        # 2. Mutations
        crs_evade_muts = int(muts['crs_evaded'].sum()) if n_muts > 0 else 0
        crs_rate_muts = (crs_evade_muts / n_muts) * 100.0 if n_muts > 0 else 0.0

        model_evade_muts = int(muts['model_evaded'].sum()) if n_muts > 0 else 0
        model_rate_muts = (model_evade_muts / n_muts) * 100.0 if n_muts > 0 else 0.0

        hybrid_evade_muts = int((muts['crs_evaded'] & muts['model_evaded']).sum()) if n_muts > 0 else 0
        hybrid_rate_muts = (hybrid_evade_muts / n_muts) * 100.0 if n_muts > 0 else 0.0

        # 3. Categorization on Mutations
        evades_crs_only = int((muts['crs_evaded'] & (~muts['model_evaded'])).sum())
        evades_model_only = int(((~muts['crs_evaded']) & muts['model_evaded']).sum())
        evades_both = int((muts['crs_evaded'] & muts['model_evaded']).sum())
        blocked_by_both = int(((~muts['crs_evaded']) & (~muts['model_evaded'])).sum())

        print(f"\n[ATTACK VECTOR: {attack_type.upper()}] (Seeds: {n_seeds:,d} | Mutation Attempts: {n_muts:,d})")
        print("-" * 105)
        print(f"{'Metric':<35s} | {'Seeds (Original)':>20s} | {'Mutations (Adversarial)':>25s}")
        print("-" * 105)
        print(f"{'CRS Evasion Rate':<35s} | {crs_evade_seeds:>6,d}/{n_seeds:<6,d} ({crs_rate_seeds:>5.1f}%) | {crs_evade_muts:>8,d}/{n_muts:<8,d} ({crs_rate_muts:>5.1f}%)")
        print(f"{'Model v2.1b Evasion Rate':<35s} | {model_evade_seeds:>6,d}/{n_seeds:<6,d} ({model_rate_seeds:>5.1f}%) | {model_evade_muts:>8,d}/{n_muts:<8,d} ({model_rate_muts:>5.1f}%)")
        print(f"{'Hybrid WAF Evasion (Evades Both)':<35s} | {hybrid_evade_seeds:>6,d}/{n_seeds:<6,d} ({hybrid_rate_seeds:>5.1f}%) | {hybrid_evade_muts:>8,d}/{n_muts:<8,d} ({hybrid_rate_muts:>5.1f}%)")
        print("-" * 105)
        print("Adversarial Mutation Categorization Breakdown:")
        print(f"  - Evades CRS Only (Caught by Model):     {evades_crs_only:>6,d} ({100.0*evades_crs_only/max(n_muts, 1):>5.1f}%)")
        print(f"  - Evades Model Only (Caught by CRS):     {evades_model_only:>6,d} ({100.0*evades_model_only/max(n_muts, 1):>5.1f}%)")
        print(f"  - Evades BOTH CRS and Model:             {evades_both:>6,d} ({100.0*evades_both/max(n_muts, 1):>5.1f}%)")
        print(f"  - Blocked by BOTH CRS and Model:         {blocked_by_both:>6,d} ({100.0*blocked_by_both/max(n_muts, 1):>5.1f}%)")
        print("-" * 105)

    print("\n" + "="*105)


if __name__ == "__main__":
    main()
