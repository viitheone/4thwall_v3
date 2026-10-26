"""
Phase D CRS Harness Controls (Item 2):
1. Smoke-test known-bad payloads that MUST return 403 at PL1.
2. Smoke-test 10 benign GETs that MUST NOT return 403.
3. Replay every SQLi seed from replay_payloads.csv in three placements:
   (a) query argument  (b) URL path  (c) POST body
   Report CRS evasion by placement.
"""

import argparse
import os
import sys
import time
import urllib.parse
import pandas as pd
import requests

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")

RATE_DELAY = 0.06  # ~16 req/s


def send_request(session, base_url, method, path, body="", ct="", timeout=5.0):
    """Send a single request and return HTTP status code."""
    if not path.startswith('/'):
        path = '/' + path
    url = urllib.parse.urljoin(base_url, path)
    headers = {"User-Agent": DEFAULT_UA}
    if ct:
        headers["Content-Type"] = ct
    try:
        time.sleep(RATE_DELAY)
        if method.upper() == "POST":
            resp = session.post(url, data=body.encode('utf-8') if body else b'',
                                headers=headers, timeout=timeout, allow_redirects=False)
        else:
            resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=False)
        return resp.status_code
    except requests.exceptions.RequestException as e:
        return 0


def main():
    parser = argparse.ArgumentParser(description="CRS Harness Control Test")
    parser.add_argument("--base_url", default="http://localhost:80")
    parser.add_argument("--replay_csv", default="outputs/replay_payloads.csv",
                        help="Path to replay_payloads.csv (for SQLi seed extraction)")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    session = requests.Session()

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: KNOWN-BAD REQUESTS (must return 403)
    # ══════════════════════════════════════════════════════════════════════
    known_bad = [
        ("SQLi (query arg OR tautology)", "GET",
         "/?id=1'+OR+'1'='1", "", ""),
        ("SQLi (UNION SELECT query arg)", "GET",
         "/?id=1+UNION+SELECT+NULL,NULL,NULL--", "", ""),
        ("XSS (<script> in query arg)", "GET",
         "/?q=<script>alert(1)</script>", "", ""),
        ("Path Traversal (../../etc/passwd)", "GET",
         "/../../etc/passwd", "", ""),
        ("Command Injection (; cat)", "GET",
         "/?cmd=test;cat+/etc/passwd", "", ""),
    ]

    print("=" * 100)
    print("PART 1: KNOWN-BAD PAYLOADS (Expected: 403 Forbidden)")
    print("=" * 100)
    print(f"{'#':<3} {'Description':<45} {'Method':<6} {'Status':>7} {'Result':<8}")
    print("-" * 100)

    all_pass_bad = True
    for i, (desc, method, path, body, ct) in enumerate(known_bad, 1):
        status = send_request(session, args.base_url, method, path, body, ct, args.timeout)
        passed = (status == 403)
        if not passed:
            all_pass_bad = False
        print(f"{i:<3} {desc:<45} {method:<6} {status:>7} {'PASS' if passed else '**FAIL**':<8}")

    print()
    if all_pass_bad:
        print("  >>> All known-bad payloads correctly blocked (403). CRS baseline VERIFIED.")
    else:
        print("  >>> WARNING: Some known-bad payloads were NOT blocked! CRS may not be in enforcement mode.")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: BENIGN REQUESTS (must NOT return 403)
    # ══════════════════════════════════════════════════════════════════════
    benign = [
        ("Homepage /", "GET", "/", "", ""),
        ("Static CSS file", "GET", "/blog/wp-includes/css/dist/block-library/style.min.css", "", ""),
        ("Static JS file", "GET", "/blog/wp-includes/js/jquery/jquery.js?ver=1.12.4", "", ""),
        ("Blog post slug", "GET", "/blog/index.php/2020/03/22/hello-world/", "", ""),
        ("Login page", "GET", "/blog/index.php/my-account/", "", ""),
        ("Password reset", "GET", "/blog/index.php/my-account/lost-password/?reset-link-sent=true", "", ""),
        ("REST API endpoint", "GET", "/blog/wp-json/wp/v2/posts?per_page=5", "", ""),
        ("Search page", "GET", "/blog/?s=lorem+ipsum", "", ""),
        ("Favicon", "GET", "/favicon.ico", "", ""),
        ("Robots.txt", "GET", "/robots.txt", "", ""),
    ]

    print("=" * 100)
    print("PART 2: BENIGN REQUESTS (Expected: NOT 403)")
    print("=" * 100)
    print(f"{'#':<3} {'Description':<55} {'Method':<6} {'Status':>7} {'Result':<8}")
    print("-" * 100)

    all_pass_benign = True
    for i, (desc, method, path, body, ct) in enumerate(benign, 1):
        status = send_request(session, args.base_url, method, path, body, ct, args.timeout)
        passed = (status != 403 and status != 0)
        if not passed:
            all_pass_benign = False
        print(f"{i:<3} {desc:<55} {method:<6} {status:>7} {'PASS' if passed else '**FAIL**':<8}")

    print()
    if all_pass_benign:
        print("  >>> All benign requests passed without blocking. CRS baseline VERIFIED.")
    else:
        print("  >>> WARNING: Some benign requests were blocked (403) or failed. Check CRS paranoia level.")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: SQLi SEED PLACEMENT VARIANTS
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("PART 3: SQLi SEED PAYLOAD PLACEMENT VARIANTS (query arg vs URL path vs POST body)")
    print("=" * 100)

    # Load SQLi seeds from replay_payloads.csv
    for cand in [args.replay_csv, os.path.join("..", args.replay_csv)]:
        if os.path.exists(cand):
            args.replay_csv = os.path.abspath(cand)
            break

    if not os.path.exists(args.replay_csv):
        print(f"  [ERROR] replay_payloads.csv not found at {args.replay_csv}")
        sys.exit(1)

    df = pd.read_csv(args.replay_csv)
    sql_df = df[df['attack_type'] == 'SQL Injection']

    # Extract unique original seeds
    seeds = sql_df.groupby('seed_id').first().reset_index()
    print(f"  Found {len(seeds)} SQLi seeds to test.\n")

    placements = {
        "query_arg":  {"blocked": 0, "evaded": 0, "error": 0, "total": 0},
        "url_path":   {"blocked": 0, "evaded": 0, "error": 0, "total": 0},
        "post_body":  {"blocked": 0, "evaded": 0, "error": 0, "total": 0},
    }

    detail_rows = []

    for _, seed in seeds.iterrows():
        sid = seed['seed_id']
        raw_payload = str(seed['original_payload'])

        # (a) Query argument placement: GET /?id=<payload>
        qa_path = "/?id=" + urllib.parse.quote(raw_payload, safe='')
        qa_status = send_request(session, args.base_url, "GET", qa_path, timeout=args.timeout)
        placements["query_arg"]["total"] += 1
        if qa_status == 403:
            placements["query_arg"]["blocked"] += 1
        elif qa_status == 0:
            placements["query_arg"]["error"] += 1
        else:
            placements["query_arg"]["evaded"] += 1

        # (b) URL path placement: GET /<payload>
        up_path = raw_payload if raw_payload.startswith('/') else '/' + raw_payload
        up_status = send_request(session, args.base_url, "GET", up_path, timeout=args.timeout)
        placements["url_path"]["total"] += 1
        if up_status == 403:
            placements["url_path"]["blocked"] += 1
        elif up_status == 0:
            placements["url_path"]["error"] += 1
        else:
            placements["url_path"]["evaded"] += 1

        # (c) POST body placement: POST / with body=<payload>
        pb_status = send_request(session, args.base_url, "POST", "/",
                                 body=f"input={urllib.parse.quote(raw_payload, safe='')}",
                                 ct="application/x-www-form-urlencoded",
                                 timeout=args.timeout)
        placements["post_body"]["total"] += 1
        if pb_status == 403:
            placements["post_body"]["blocked"] += 1
        elif pb_status == 0:
            placements["post_body"]["error"] += 1
        else:
            placements["post_body"]["evaded"] += 1

        detail_rows.append({
            "seed_id": sid,
            "payload": raw_payload[:80],
            "query_arg_status": qa_status,
            "url_path_status": up_status,
            "post_body_status": pb_status,
        })

    # Print detail table
    print(f"  {'Seed':<12} {'QueryArg':>10} {'URLPath':>10} {'POSTBody':>10} {'Payload (trunc)':<60}")
    print("  " + "-" * 110)
    for r in detail_rows:
        qa_s = str(r['query_arg_status'])
        up_s = str(r['url_path_status'])
        pb_s = str(r['post_body_status'])
        print(f"  {r['seed_id']:<12} {qa_s:>10} {up_s:>10} {pb_s:>10} {r['payload']:<60}")

    # Summary table
    print("\n  PLACEMENT SUMMARY:")
    print(f"  {'Placement':<20} {'Total':>6} {'Blocked (403)':>14} {'Evaded':>10} {'Error':>8} {'CRS Evasion Rate':>18}")
    print("  " + "-" * 80)
    for name, stats in placements.items():
        total = stats['total']
        blocked = stats['blocked']
        evaded = stats['evaded']
        err = stats['error']
        rate = f"{100.0 * evaded / total:.1f}%" if total > 0 else "N/A"
        print(f"  {name:<20} {total:>6} {blocked:>14} {evaded:>10} {err:>8} {rate:>18}")

    print()
    print("=" * 100)
    print("HARNESS CONTROLS COMPLETE.")
    print("=" * 100)


if __name__ == "__main__":
    main()
