"""
Compare PL1 vs PL2 on Structural Variants (Item 4)
"""

import os
import pandas as pd
import numpy as np

def evaluate_file(csv_path, pl_name):
    df = pd.read_csv(csv_path)
    df['crs_blocked'] = (df['http_status'] == 403)
    df['crs_evaded'] = (df['http_status'] != 403) & (df['http_status'] != 0)
    df['model_evaded'] = df['model_evaded'].astype(bool)
    df['both_evaded'] = df['crs_evaded'] & df['model_evaded']
    df['blocked_by_both'] = df['crs_blocked'] & (~df['model_evaded'])
    df['crs_only_evaded'] = df['crs_evaded'] & (~df['model_evaded'])
    df['model_only_evaded'] = df['crs_blocked'] & df['model_evaded']

    def get_base_placement(row):
        vt = row['variant_type']
        pl = row['placement']
        if vt == 'placement_query_arg': return 'Query Arg (?id=...)'
        elif vt == 'placement_url_path': return 'URL Path (/...)'
        elif vt in ('placement_post_body_no_ct', 'body_truncation_padding_600'): return 'POST Body'
        elif vt == 'path_token_overflow_256': return 'URL Path (/...)'
        elif vt == 'header_cookie': return 'Header (Cookie)'
        elif vt == 'header_referer': return 'Header (Referer)'
        elif vt == 'header_x-forwarded-for': return 'Header (X-Forwarded-For)'
        elif vt == 'ua_mutation_inject': return 'Header (User-Agent)'
        elif pl == 'body': return 'POST Body'
        else: return 'URL Path (/...)'

    df['base_placement'] = df.apply(get_base_placement, axis=1)

    controls = df[df['variant_type'].isin(['placement_query_arg', 'placement_post_body_no_ct', 'placement_url_path'])].copy()
    seed_ctrl_map = {}
    for _, r in controls.iterrows():
        sid = r['seed_id']
        vt = r['variant_type']
        if sid not in seed_ctrl_map: seed_ctrl_map[sid] = {}
        seed_ctrl_map[sid][vt] = r['crs_blocked']

    def is_seed_blocked_in_base(row):
        sid = row['seed_id']
        bp = row['base_placement']
        cmap = seed_ctrl_map.get(sid, {})
        if 'Query Arg' in bp: return cmap.get('placement_query_arg', False)
        elif 'POST Body' in bp: return cmap.get('placement_post_body_no_ct', False)
        elif 'URL Path' in bp: return cmap.get('placement_url_path', False)
        elif 'Header' in bp: return cmap.get('placement_query_arg', False)
        return False

    df['seed_blocked_in_base'] = df.apply(is_seed_blocked_in_base, axis=1)

    results = {}
    for attack_name in ["SQL Injection", "Code Injection"]:
        sub = df[df['attack_type'] == attack_name]
        tot = len(sub)
        
        # All
        both_bk = sub['blocked_by_both'].sum()
        crs_only = sub['crs_only_evaded'].sum()
        mod_only = sub['model_only_evaded'].sum()
        both_ev = sub['both_evaded'].sum()

        # Filtered
        sub_cond = sub[sub['seed_blocked_in_base'] == True]
        tot_c = len(sub_cond)
        both_bk_c = sub_cond['blocked_by_both'].sum()
        crs_only_c = sub_cond['crs_only_evaded'].sum()
        mod_only_c = sub_cond['model_only_evaded'].sum()
        both_ev_c = sub_cond['both_evaded'].sum()

        results[attack_name] = {
            "all": (tot, both_bk, crs_only, mod_only, both_ev),
            "cond": (tot_c, both_bk_c, crs_only_c, mod_only_c, both_ev_c)
        }
    return results

def main():
    f_pl1 = "outputs/crs_results_structural2_thresh5_pl1.csv"
    f_pl2 = "outputs/crs_results_structural2_thresh5_pl2.csv"

    if not os.path.exists(f_pl2):
        print(f"Waiting for {f_pl2}...")
        return

    res1 = evaluate_file(f_pl1, "PL1")
    res2 = evaluate_file(f_pl2, "PL2")

    print("=" * 110)
    print("CRS PARANOIA LEVEL COMPARISON: PL1 vs PL2 @ DEFAULT THRESHOLD 5")
    print("=" * 110)

    for atk in ["SQL Injection", "Code Injection"]:
        print(f"\n>>> ATTACK CLASS: {atk.upper()}")
        print(f"--- (i) All 1,700 Variants ---")
        t1, bb1, co1, mo1, be1 = res1[atk]["all"]
        t2, bb2, co2, mo2, be2 = res2[atk]["all"]

        print(f"  {'Metric':<40} | {'PL1 (Threshold 5)':>25} | {'PL2 (Threshold 5)':>25}")
        print("  " + "-" * 95)
        print(f"  {'Blocked by BOTH CRS & Model':<40} | {bb1:>6} ({100*bb1/t1:>5.1f}%) | {bb2:>6} ({100*bb2/t2:>5.1f}%)")
        print(f"  {'Evades CRS Only (Caught by Model)':<40} | {co1:>6} ({100*co1/t1:>5.1f}%) | {co2:>6} ({100*co2/t2:>5.1f}%)")
        print(f"  {'Evades Model Only (Caught by CRS)':<40} | {mo1:>6} ({100*mo1/t1:>5.1f}%) | {mo2:>6} ({100*mo2/t2:>5.1f}%)")
        print(f"  {'Evades BOTH CRS and Model':<40} | {be1:>6} ({100*be1/t1:>5.1f}%) | {be2:>6} ({100*be2/t2:>5.1f}%)")

        print(f"\n--- (ii) Filtered: Unmodified Seed Blocked by CRS ---")
        t1_c, bb1_c, co1_c, mo1_c, be1_c = res1[atk]["cond"]
        t2_c, bb2_c, co2_c, mo2_c, be2_c = res2[atk]["cond"]

        print(f"  Retained Variants: PL1={t1_c:,d} / 1,700 | PL2={t2_c:,d} / 1,700")
        print(f"  {'Metric':<40} | {'PL1 (Filtered)':>25} | {'PL2 (Filtered)':>25}")
        print("  " + "-" * 95)
        print(f"  {'Blocked by BOTH CRS & Model':<40} | {bb1_c:>6} ({100*bb1_c/t1_c:>5.1f}%) | {bb2_c:>6} ({100*bb2_c/t2_c:>5.1f}%)")
        print(f"  {'Evades CRS Only (Caught by Model)':<40} | {co1_c:>6} ({100*co1_c/t1_c:>5.1f}%) | {co2_c:>6} ({100*co2_c/t2_c:>5.1f}%)")
        print(f"  {'Evades Model Only (Caught by CRS)':<40} | {mo1_c:>6} ({100*mo1_c/t1_c:>5.1f}%) | {mo2_c:>6} ({100*mo2_c/t2_c:>5.1f}%)")
        print(f"  {'Evades BOTH CRS and Model':<40} | {be1_c:>6} ({100*be1_c/t1_c:>5.1f}%) | {be2_c:>6} ({100*be2_c/t2_c:>5.1f}%)")

    print("\n" + "=" * 110)

if __name__ == "__main__":
    main()
