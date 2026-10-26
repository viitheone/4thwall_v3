"""
Replay Payloads against ModSecurity / OWASP Core Rule Set (CRS):
Takes outputs/replay_payloads.csv and replays each request against a live WAF endpoint
(default http://localhost:8080) with rate limiting (20 req/s) and timeout.
Records HTTP status and outputs crs_results.csv.
"""

import argparse
import os
import sys
import time
import urllib.parse
import pandas as pd
import requests

DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
RATE_LIMIT_DELAY = 1.0 / 20.0  # 20 requests per second = 50ms delay


def main():
    parser = argparse.ArgumentParser(description="Replay Payloads against ModSecurity / OWASP CRS")
    parser.add_argument("--input_csv", default="outputs/replay_payloads.csv",
                        help="Path to generated replay_payloads.csv")
    parser.add_argument("--output_csv", default="outputs/crs_results.csv",
                        help="Path to output crs_results.csv")
    parser.add_argument("--base_url", default="http://localhost:8080",
                        help="Base URL of ModSecurity CRS target (default: http://localhost:8080)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="HTTP request timeout in seconds (default: 5.0)")
    parser.add_argument("--rate_limit", type=float, default=20.0,
                        help="Max requests per second (default: 20.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"[ERROR] Input CSV not found: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,d} requests from {args.input_csv}.")
    print(f"Target Base URL: {args.base_url}")
    print(f"Rate Limit: {args.rate_limit} req/s (min interval: {1.0/args.rate_limit:.3f}s)")

    min_interval = 1.0 / args.rate_limit if args.rate_limit > 0 else 0
    results = []

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})

    start_time = time.time()
    last_req_time = 0.0

    print("\nReplaying requests against CRS...")
    for idx, row in df.iterrows():
        method = str(row.get('http_method', 'GET')).upper()
        url_path = str(row.get('url_path_with_payload', '/'))
        if not url_path.startswith('/'):
            url_path = '/' + url_path

        body = row.get('body', '')
        if pd.isna(body):
            body = ''
        else:
            body = str(body)

        ct = row.get('content_type', '')
        ua = row.get('user_agent', '')
        if pd.isna(ua) or not str(ua).strip() or str(ua).lower() == 'nan':
            ua = DEFAULT_UA
        else:
            ua = str(ua).strip()

        headers = {"User-Agent": ua}
        if pd.notna(ct) and str(ct).strip() and str(ct).lower() != 'nan':
            headers["Content-Type"] = str(ct).strip()

        h_name = row.get('header_name', '')
        h_val = row.get('header_value', '')
        if pd.notna(h_name) and pd.notna(h_val) and str(h_name).strip() and str(h_name).lower() != 'nan':
            headers[str(h_name).strip()] = str(h_val).strip()


        # Enforce rate limit
        now = time.time()
        elapsed_since_last = now - last_req_time
        if elapsed_since_last < min_interval:
            time.sleep(min_interval - elapsed_since_last)

        target_url = urllib.parse.urljoin(args.base_url, url_path)
        status_code = None
        error_msg = ""
        last_req_time = time.time()

        try:
            if method == 'POST':
                resp = session.post(target_url, data=body.encode('utf-8'), headers=headers,
                                    timeout=args.timeout, allow_redirects=False)
            else:
                resp = session.get(target_url, headers=headers,
                                   timeout=args.timeout, allow_redirects=False)
            status_code = resp.status_code
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            status_code = 0

        # In ModSecurity CRS, 403 Forbidden indicates a rule trigger (blocked)
        is_blocked = (status_code == 403)
        # Evasion occurs when an attack request is NOT blocked (status != 403 and no connection error)
        is_evaded = (status_code != 403 and status_code != 0)

        row_dict = row.to_dict()
        row_dict["http_status"] = status_code
        row_dict["crs_blocked"] = is_blocked
        row_dict["crs_evaded"] = is_evaded
        row_dict["crs_error"] = error_msg
        results.append(row_dict)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            rate = (idx + 1) / (time.time() - start_time)
            print(f"  [{idx+1:,d}/{len(df):,d}] Last Status: {status_code} | Speed: {rate:.1f} req/s")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\n[DONE] Saved CRS replay results to {args.output_csv}")

    # Summary
    n_total = len(out_df)
    n_blocked = int(out_df['crs_blocked'].sum())
    n_evaded = int(out_df['crs_evaded'].sum())
    n_err = int((out_df['http_status'] == 0).sum())
    print("\n" + "="*60)
    print("CRS REPLAY SUMMARY")
    print("="*60)
    print(f"Total Requests Replayed: {n_total:,d}")
    print(f"Blocked by CRS (403):    {n_blocked:,d} ({100.0*n_blocked/n_total:.1f}%)")
    print(f"Evaded CRS (non-403):    {n_evaded:,d} ({100.0*n_evaded/n_total:.1f}%)")
    if n_err > 0:
        print(f"Connection Errors:       {n_err:,d} ({100.0*n_err/n_total:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
