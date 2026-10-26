"""
Phase D Search Audit (Item 1):
Analyses outputs/replay_payloads.csv from the WAF-A-MoLE run.
Reports per-attack-type:
  - candidates generated, dropped by validity, scored
  - per-seed: min model score reached vs frozen threshold, score drop
  - mean score of seeds vs mutants
  - flags seeds with score drops > 0.3
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser(description="Phase D Search Audit")
    parser.add_argument("--replay_csv", default="outputs/replay_payloads.csv")
    parser.add_argument("--thresholds_json", default="outputs/frozen_v2_1b/thresholds.json")
    args = parser.parse_args()

    # resolve paths
    for cand_base in [args.replay_csv, os.path.join("..", args.replay_csv)]:
        if os.path.exists(cand_base):
            args.replay_csv = os.path.abspath(cand_base)
            break
    for cand_base in [args.thresholds_json, os.path.join("..", args.thresholds_json)]:
        if os.path.exists(cand_base):
            args.thresholds_json = os.path.abspath(cand_base)
            break

    with open(args.thresholds_json, "r") as f:
        thresholds = json.load(f)

    df = pd.read_csv(args.replay_csv)
    print(f"Loaded {len(df):,d} records from {args.replay_csv}\n")

    # Identify seeds: first row per seed_id
    first_per_seed = df.groupby('seed_id').first().reset_index()
    is_seed = df['original_payload'] == df['mutated_payload']

    for attack_type in df['attack_type'].unique():
        sub = df[df['attack_type'] == attack_type].copy()
        seeds_sub = sub[sub['original_payload'] == sub['mutated_payload']]
        muts_sub = sub[sub['original_payload'] != sub['mutated_payload']]

        # Threshold for target label
        if 'SQL' in attack_type:
            target_label = '66 - SQL Injection'
        elif 'Code' in attack_type:
            target_label = '242 - Code Injection'
        else:
            target_label = attack_type
        frozen_thresh = thresholds.get(target_label, 0.5)

        # Validity filtering (only meaningful for SQLi)
        if 'sqlparse_valid' in sub.columns:
            invalid_count = int((muts_sub['sqlparse_valid'] == False).sum())
        else:
            invalid_count = 0

        n_seeds = len(sub['seed_id'].unique())
        n_total = len(sub)
        n_mut_rows = len(muts_sub)
        n_seed_rows = len(seeds_sub)

        print("=" * 110)
        print(f"SEARCH AUDIT: {attack_type.upper()}")
        print("=" * 110)
        print(f"  Seeds Sampled:                    {n_seeds}")
        print(f"  Total Records (seeds + mutants):  {n_total:,d}")
        print(f"  Seed Records:                     {n_seed_rows}")
        print(f"  Mutation Records:                 {n_mut_rows:,d}")
        print(f"  Dropped by Validity Filter:       {invalid_count:,d}")
        print(f"  Scored (all records pass model):   {n_total:,d}")
        print(f"  Frozen Threshold (T):             {frozen_thresh}")
        print()

        # Per-seed analysis
        print(f"  {'Seed ID':<12} {'Orig Score':>11} {'Min Mutant':>11} {'Max Drop':>10} {'Thresh':>7} {'Evaded?':>8} {'Muts':>5} {'Flag':>6}")
        print("  " + "-" * 85)

        flagged = []
        seed_orig_scores = []
        all_mut_scores = []

        for sid in sorted(sub['seed_id'].unique()):
            sid_rows = sub[sub['seed_id'] == sid]
            seed_row = sid_rows[sid_rows['original_payload'] == sid_rows['mutated_payload']]
            mut_rows = sid_rows[sid_rows['original_payload'] != sid_rows['mutated_payload']]

            if len(seed_row) > 0:
                orig_score = float(seed_row.iloc[0]['model_score_original'])
            else:
                orig_score = float(sid_rows.iloc[0]['model_score_original'])

            seed_orig_scores.append(orig_score)

            if len(mut_rows) > 0:
                min_score = float(mut_rows['model_score_mutated'].min())
                all_mut_scores.extend(mut_rows['model_score_mutated'].tolist())
            else:
                min_score = orig_score

            drop = orig_score - min_score
            any_evade = bool(sid_rows['model_evaded'].any())
            n_muts = len(mut_rows)
            flag = ">>FLAG" if drop > 0.3 else ""

            if flag:
                flagged.append((sid, orig_score, min_score, drop))

            print(f"  {sid:<12} {orig_score:>11.4f} {min_score:>11.4f} {drop:>10.4f} {frozen_thresh:>7.2f} {'YES' if any_evade else 'no':>8} {n_muts:>5} {flag:>6}")

        print()
        mean_seed = float(np.mean(seed_orig_scores)) if seed_orig_scores else 0
        mean_mut = float(np.mean(all_mut_scores)) if all_mut_scores else 0
        std_mut = float(np.std(all_mut_scores)) if all_mut_scores else 0
        min_mut = float(np.min(all_mut_scores)) if all_mut_scores else 0
        max_mut = float(np.max(all_mut_scores)) if all_mut_scores else 0

        print(f"  Mean Seed Score:               {mean_seed:.4f}")
        print(f"  Mean Mutant Score:             {mean_mut:.4f}  (std: {std_mut:.4f})")
        print(f"  Min Mutant Score:              {min_mut:.4f}")
        print(f"  Max Mutant Score:              {max_mut:.4f}")
        print(f"  Score Gap (Mean Seed - Mean Mut): {mean_seed - mean_mut:.4f}")
        print()

        if flagged:
            print(f"  *** FLAGGED SEEDS (score drop > 0.3): {len(flagged)} ***")
            for sid, os_, ms, dr in flagged:
                print(f"      {sid}: {os_:.4f} -> {ms:.4f} (drop {dr:.4f})")
        else:
            print(f"  *** No seeds flagged (max score drop <= 0.3). ***")
        print()

    print("=" * 110)
    print("SEARCH AUDIT COMPLETE.")
    print("=" * 110)


if __name__ == "__main__":
    main()
