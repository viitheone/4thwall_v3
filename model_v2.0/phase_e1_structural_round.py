"""
Phase E1 Item (b): Structural Round Detailed Analysis
- Print all 17 transformations with variants, CRS evaded, model evaded, both evaded, per attack type, and BASE placement.
- Add control rows: unmodified seed in query arg and in POST body.
- Recompute 'CRS only / model only / both':
  (i) on all variants
  (ii) only on variants whose unmodified seed in the same base placement is BLOCKED by CRS.
"""

import pandas as pd
import numpy as np
import os
import sys

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="outputs/crs_results_structural2_thresh5_pl1.csv")
    args = parser.parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        csv_path = "outputs/crs_results_structural2.csv"
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,d} records from {csv_path}\n")

    # Boolean columns
    df['crs_blocked'] = (df['http_status'] == 403)
    df['crs_evaded'] = (df['http_status'] != 403) & (df['http_status'] != 0)
    df['model_evaded'] = df['model_evaded'].astype(bool)
    df['both_evaded'] = df['crs_evaded'] & df['model_evaded']
    df['blocked_by_both'] = df['crs_blocked'] & (~df['model_evaded'])
    df['crs_only_evaded'] = df['crs_evaded'] & (~df['model_evaded'])
    df['model_only_evaded'] = df['crs_blocked'] & df['model_evaded']

    # Define base placement mapping for each variant type
    # For encoding and ua_swap variants, base placement was original location (path or body).
    # In replay_payloads2, placement column records 'path', 'body', or 'header'.
    # Specifically:
    # placement_query_arg: query_arg
    # placement_url_path: url_path
    # placement_post_body_no_ct: post_body
    # body_truncation_padding_600: post_body
    # path_token_overflow_256: url_path
    # header_cookie: header_cookie
    # header_referer: header_referer
    # header_x-forwarded-for: header_xff
    # ua_mutation_inject: header_ua
    
    def get_base_placement(row):
        vt = row['variant_type']
        pl = row['placement']
        if vt == 'placement_query_arg':
            return 'Query Arg (?id=...)'
        elif vt == 'placement_url_path':
            return 'URL Path (/...)'
        elif vt in ('placement_post_body_no_ct', 'body_truncation_padding_600'):
            return 'POST Body'
        elif vt == 'path_token_overflow_256':
            return 'URL Path (/...)'
        elif vt == 'header_cookie':
            return 'Header (Cookie)'
        elif vt == 'header_referer':
            return 'Header (Referer)'
        elif vt == 'header_x-forwarded-for':
            return 'Header (X-Forwarded-For)'
        elif vt == 'ua_mutation_inject':
            return 'Header (User-Agent)'
        elif pl == 'body':
            return 'POST Body'
        else:
            return 'URL Path (/...)'

    df['base_placement'] = df.apply(get_base_placement, axis=1)

    # Find control block status per seed:
    # Control 1: Query Arg (placement_query_arg)
    # Control 2: POST Body (placement_post_body_no_ct)
    # Control 3: URL Path (placement_url_path)
    
    controls = df[df['variant_type'].isin(['placement_query_arg', 'placement_post_body_no_ct', 'placement_url_path'])].copy()
    
    # Map seed_id -> {placement_type: crs_blocked}
    seed_ctrl_map = {}
    for _, r in controls.iterrows():
        sid = r['seed_id']
        vt = r['variant_type']
        if sid not in seed_ctrl_map:
            seed_ctrl_map[sid] = {}
        seed_ctrl_map[sid][vt] = r['crs_blocked']

    # For condition (ii): "only on variants whose unmodified seed in the same base placement is BLOCKED by CRS"
    # What is the unmodified seed's status in the corresponding placement?
    def is_seed_blocked_in_base_placement(row):
        sid = row['seed_id']
        bp = row['base_placement']
        cmap = seed_ctrl_map.get(sid, {})
        if 'Query Arg' in bp:
            return cmap.get('placement_query_arg', False)
        elif 'POST Body' in bp:
            return cmap.get('placement_post_body_no_ct', False)
        elif 'URL Path' in bp:
            return cmap.get('placement_url_path', False)
        elif 'Header' in bp:
            # For header variants, the seed payload itself was tested in query arg or post body;
            # if testing against Query Arg baseline:
            return cmap.get('placement_query_arg', False)
        return False

    df['seed_blocked_in_base'] = df.apply(is_seed_blocked_in_base_placement, axis=1)

    for attack_name in df['attack_type'].unique():
        sub = df[df['attack_type'] == attack_name].copy()
        n_seeds = len(sub['seed_id'].unique())
        
        print("=" * 130)
        print(f"PHASE E1 (b): STRUCTURAL ROUND BREAKDOWN - {attack_name.upper()} ({n_seeds} seeds)")
        print("=" * 130)

        # 1. Print all 17 transformations + control rows
        print(f"{'#':<3} {'Transformation / Variant':<32} {'Base Placement':<26} {'Total':>5} {'CRS Ev':>7} {'Mod Ev':>7} {'Both Ev':>8} {'CRS Blk':>8}")
        print("-" * 130)

        # Control rows first
        qa_ctrl = sub[sub['variant_type'] == 'placement_query_arg']
        body_ctrl = sub[sub['variant_type'] == 'placement_post_body_no_ct']
        path_ctrl = sub[sub['variant_type'] == 'placement_url_path']

        print(f"[C1] {'CONTROL: Seed in Query Arg':<31} {'Query Arg (?id=...)':<26} {len(qa_ctrl):>5} {qa_ctrl['crs_evaded'].sum():>7} {qa_ctrl['model_evaded'].sum():>7} {qa_ctrl['both_evaded'].sum():>8} {qa_ctrl['crs_blocked'].sum():>8}")
        print(f"[C2] {'CONTROL: Seed in POST Body':<31} {'POST Body':<26} {len(body_ctrl):>5} {body_ctrl['crs_evaded'].sum():>7} {body_ctrl['model_evaded'].sum():>7} {body_ctrl['both_evaded'].sum():>8} {body_ctrl['crs_blocked'].sum():>8}")
        print(f"[C3] {'CONTROL: Seed in URL Path':<31} {'URL Path (/...)':<26} {len(path_ctrl):>5} {path_ctrl['crs_evaded'].sum():>7} {path_ctrl['model_evaded'].sum():>7} {path_ctrl['both_evaded'].sum():>8} {path_ctrl['crs_blocked'].sum():>8}")
        print("-" * 130)

        idx = 1
        for vtype in sorted(sub['variant_type'].unique()):
            vdf = sub[sub['variant_type'] == vtype]
            tot = len(vdf)
            bp = vdf['base_placement'].iloc[0] if vdf['base_placement'].nunique() == 1 else "Mixed (Path/Body)"
            crs_ev = vdf['crs_evaded'].sum()
            mod_ev = vdf['model_evaded'].sum()
            both_ev = vdf['both_evaded'].sum()
            crs_bk = vdf['crs_blocked'].sum()
            print(f"{idx:<3} {vtype:<32} {bp:<26} {tot:>5} {crs_ev:>7} {mod_ev:>7} {both_ev:>8} {crs_bk:>8}")
            idx += 1

        print("-" * 130)

        # 2. Recompute Defense-in-Depth Matrix
        # (i) On ALL variants
        tot_all = len(sub)
        crs_only_all = sub['crs_only_evaded'].sum()
        mod_only_all = sub['model_only_evaded'].sum()
        both_ev_all = sub['both_evaded'].sum()
        both_bk_all = sub['blocked_by_both'].sum()

        print(f"\n(i) DEFENSE-IN-DEPTH MATRIX ON ALL VARIANTS ({tot_all:,d} requests):")
        print(f"    - Blocked by BOTH CRS & Model:          {both_bk_all:>6} ({100.0*both_bk_all/tot_all:>5.1f}%)")
        print(f"    - Evades CRS Only (Caught by Model):    {crs_only_all:>6} ({100.0*crs_only_all/tot_all:>5.1f}%)")
        print(f"    - Evades Model Only (Caught by CRS):    {mod_only_all:>6} ({100.0*mod_only_all/tot_all:>5.1f}%)")
        print(f"    - Evades BOTH CRS and Model:            {both_ev_all:>6} ({100.0*both_ev_all/tot_all:>5.1f}%)")

        # (ii) Only on variants whose unmodified seed in the same base placement is BLOCKED by CRS
        sub_cond = sub[sub['seed_blocked_in_base'] == True]
        tot_cond = len(sub_cond)
        crs_only_cond = sub_cond['crs_only_evaded'].sum()
        mod_only_cond = sub_cond['model_only_evaded'].sum()
        both_ev_cond = sub_cond['both_evaded'].sum()
        both_bk_cond = sub_cond['blocked_by_both'].sum()

        print(f"\n(ii) DEFENSE-IN-DEPTH MATRIX FILTERED TO VARIANTS WHOSE UNMODIFIED SEED IS BLOCKED BY CRS:")
        print(f"    (Retained {tot_cond:,d} / {tot_all:,d} variants whose base seed is blocked by CRS)")
        if tot_cond > 0:
            print(f"    - Blocked by BOTH CRS & Model:          {both_bk_cond:>6} ({100.0*both_bk_cond/tot_cond:>5.1f}%)")
            print(f"    - Evades CRS Only (Caught by Model):    {crs_only_cond:>6} ({100.0*crs_only_cond/tot_cond:>5.1f}%)")
            print(f"    - Evades Model Only (Caught by CRS):    {mod_only_cond:>6} ({100.0*mod_only_cond/tot_cond:>5.1f}%)")
            print(f"    - Evades BOTH CRS and Model:            {both_ev_cond:>6} ({100.0*both_ev_cond/tot_cond:>5.1f}%)")
        print()

if __name__ == "__main__":
    main()
