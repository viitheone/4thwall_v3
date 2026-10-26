"""
Comparative Evaluation for Structural Evasion (Item 3):
Evaluates outputs/crs_results_structural2.csv to compare:
- CRS Evasion Rate vs Model v2.1b Evasion Rate across structural variants:
  * Placement: query arg, url path, post body (CT-less)
  * Body Truncation Padding: 600-char benign prefix
  * Token Overflow Padding: > 256 tokens in path
  * Header Placement (unserialized): Cookie, Referer, X-Forwarded-For
  * Encodings: Double URL, Mixed Case, Unicode Escapes
  * UA Variations: Benign UA swaps, UA mutation inject
- Hybrid WAF Evasion Rate (evades BOTH CRS and Model)
- Breakdown: Evades CRS Only, Evades Model Only, Evades Both, Blocked by Both
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate Structural Evasion: CRS vs Model v2.1b")
    parser.add_argument("--input_csv", default="outputs/crs_results_structural2.csv",
                        help="Path to crs_results_structural2.csv produced by replay_crs.py")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"[ERROR] Input CSV not found: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,d} records from {args.input_csv}.\n")

    # Ensure boolean columns
    df['crs_evaded'] = (df['http_status'] != 403) & (df['http_status'] != 0)
    df['crs_blocked'] = (df['http_status'] == 403)
    df['model_evaded'] = df['model_evaded'].astype(bool)
    df['hybrid_evaded'] = df['crs_evaded'] & df['model_evaded']

    categories = {
        "Placement": [
            "placement_query_arg", "placement_url_path", "placement_post_body_no_ct"
        ],
        "Length Limits (Padding)": [
            "body_truncation_padding_600", "path_token_overflow_256"
        ],
        "Unserialized Headers": [
            "header_cookie", "header_referer", "header_x-forwarded-for"
        ],
        "Encodings": [
            "double_url_encoding", "mixed_case_encoding", "unicode_escapes"
        ],
        "UA Variations": [
            "ua_swap_0", "ua_swap_1", "ua_swap_2", "ua_swap_3", "ua_swap_4", "ua_mutation_inject"
        ],
    }

    print("=" * 115)
    print("STRUCTURAL EVASION: CRS vs MODEL v2.1b COMPARATIVE ANALYSIS")
    print("=" * 115)

    for atk in df['attack_type'].unique():
        sub = df[df['attack_type'] == atk].copy()
        n_total = len(sub)
        print(f"\n=========================================================================================")
        print(f" ATTACK VECTOR: {atk.upper()} (Total Structural Variants: {n_total:,d})")
        print(f"=========================================================================================")

        # Table 1: Per-Variant Breakdown
        print(f"\n  {'Variant Type':<32} {'Total':>6} | {'CRS Evaded':>12} | {'Model Evaded':>13} | {'Hybrid Evaded':>14}")
        print("  " + "-" * 85)

        for vtype in sorted(sub['variant_type'].unique()):
            vdf = sub[sub['variant_type'] == vtype]
            tot = len(vdf)
            crs_ev = int(vdf['crs_evaded'].sum())
            mod_ev = int(vdf['model_evaded'].sum())
            hyb_ev = int(vdf['hybrid_evaded'].sum())

            crs_pct = f"{100.0 * crs_ev / tot:.1f}%" if tot > 0 else "N/A"
            mod_pct = f"{100.0 * mod_ev / tot:.1f}%" if tot > 0 else "N/A"
            hyb_pct = f"{100.0 * hyb_ev / tot:.1f}%" if tot > 0 else "N/A"

            print(f"  {vtype:<32} {tot:>6} | {crs_ev:>5} ({crs_pct:>5}) | {mod_ev:>6} ({mod_pct:>5}) | {hyb_ev:>7} ({hyb_pct:>5})")

        # Table 2: Category Roll-Up
        print(f"\n  CATEGORY ROLL-UP:")
        print(f"  {'Category':<32} {'Total':>6} | {'CRS Evaded':>12} | {'Model Evaded':>13} | {'Hybrid Evaded':>14}")
        print("  " + "-" * 85)

        for cat_name, vkeys in categories.items():
            cat_df = sub[sub['variant_type'].isin(vkeys)]
            tot = len(cat_df)
            if tot == 0:
                continue
            crs_ev = int(cat_df['crs_evaded'].sum())
            mod_ev = int(cat_df['model_evaded'].sum())
            hyb_ev = int(cat_df['hybrid_evaded'].sum())

            crs_pct = f"{100.0 * crs_ev / tot:.1f}%"
            mod_pct = f"{100.0 * mod_ev / tot:.1f}%"
            hyb_pct = f"{100.0 * hyb_ev / tot:.1f}%"

            print(f"  {cat_name:<32} {tot:>6} | {crs_ev:>5} ({crs_pct:>5}) | {mod_ev:>6} ({mod_pct:>5}) | {hyb_ev:>7} ({hyb_pct:>5})")

        # Table 3: Overall Categorization
        crs_only = int((sub['crs_evaded'] & (~sub['model_evaded'])).sum())
        mod_only = int(((~sub['crs_evaded']) & sub['model_evaded']).sum())
        both_ev = int((sub['crs_evaded'] & sub['model_evaded']).sum())
        both_bk = int(((~sub['crs_evaded']) & (~sub['model_evaded'])).sum())

        print(f"\n  DEFENSE-IN-DEPTH MATRIX ({atk.upper()}):")
        print(f"    - Blocked by BOTH CRS & Model:           {both_bk:>6} ({100.0*both_bk/n_total:>5.1f}%)")
        print(f"    - Evades CRS Only (Caught by Model):     {crs_only:>6} ({100.0*crs_only/n_total:>5.1f}%)")
        print(f"    - Evades Model Only (Caught by CRS):     {mod_only:>6} ({100.0*mod_only/n_total:>5.1f}%)")
        print(f"    - Evades BOTH CRS and Model:             {both_ev:>6} ({100.0*both_ev/n_total:>5.1f}%)")

    print("\n" + "=" * 115)


if __name__ == "__main__":
    main()
