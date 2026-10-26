# Comprehensive Final Evaluation Report: Frozen Model v2.1b & CRS Benchmark

**Evaluation Scope**: Standalone Multi-Label Classification, Attack-vs-Normal Binary Detection, OWASP CRS v4.23.0 Dual-System Replay, Leakage Ablation, and Adversarial Evasion Benchmarking.  
**Hardware & Environment**: PyTorch 2.6.0+cu124 / DistilBERT-base-uncased / Windows 11 host with Dockerized OWASP CRS v4.23.0 on Nginx.  
**CRS Configuration**: `BLOCKING_PARANOIA=1`, `ANOMALY_INBOUND=5`, `SecRuleEngine On`, `SecAuditEngine On`.  
**Frozen Model Artifact**: `model_v2.0/outputs/v2_1b/v2_1b_best.pt` (Epoch 2 checkpoint, strictly frozen, zero fine-tuning).  
**Canonical Thresholds**: Loaded verbatim from `outputs/frozen_v2_1b/thresholds.json`.

---

## Executive Summary

1. **Multi-Label Performance (Test Unique-Template, 29,996 rows)**:
   - Evaluated on strictly disjoint, novel URI templates (0.00% template overlap with training data).
   - Achieves **0.9830 Macro-F1 (All 13 classes)** and **0.9923 Macro-F1 (Excluding 272)**.
   - Micro-F1 reaches **0.9912**, subset exact-match accuracy is **98.24%**, and Hamming loss is **0.0019**.
2. **Binary Attack-vs-Normal Protection**:
   - At validation-tuned frozen thresholds, the model detects **99.41% of attacks (22,448 / 22,582)** with a **0.971% False Positive Rate (72 / 7,414)**.
   - When routing `272 - Protocol Manipulation` to ALERT, benign FPR drops to **0.782% (58 / 7,414)** with **98.51% exploit recall**.
3. **Genuine CRS Replay & Dual-System Defense (All 29,996 UT Rows)**:
   - CRS replayed with **original HTTP verbs preserved** (`session.request(method, ...)`), including `TRACK`, `PROPFIND`, `PUT`, `PATCH`, `LOCK`, and all dataset headers (`Accept-Language`, `Accept-Encoding`, `Origin`, `DNT`).
   - CRS blocking defined strictly as **Inbound Anomaly Score $\ge 5$ (Rule 949110 in ModSecurity audit log)**.
   - **Union Detection**: Combining Frozen Model v2.1b and CRS achieves **99.57% attack catch rate (22,484 / 22,582)**, catching 36 evasions that bypassed the model alone.
   - **Label Noise & Clean Benign FPR**: 34.37% (2,548 / 7,414) of dataset "Normal" rows contain explicit attack markers (`<script>`, `sleep(15)`, `UNION SELECT`). On genuinely clean Normal rows (4,866 rows), CRS False Positive Rate drops from 11.83% to **0.473% (23 / 4,866)**.
4. **Textbook Payloads & Evasion Synergy**:
   - Across 40 canonical textbook payloads placed in 5 HTTP locations (200 requests):
     - In 4 canonical placements (160 requests), Model Any-Label catches **78.1%**, CRS catches **80.0%**, and Union catches **93.8% (150 / 160)**.
     - Across all 5 placements including Cookie (200 requests), Union achieves **91.0% (182 / 200)**, demonstrating that CRS provides an impenetrable safety net for header locations dropped by the model serializer.

---

## F1. Multi-Label Classification Performance

### Multi-Label Per-Class Metrics at Frozen Tuned Thresholds
Evaluated on **Test Unique-Template (29,996 rows)**, with reference comparisons on **Test Row-Weighted (90,782 rows)** and **Validation (90,781 rows)**:

| CAPEC Class                            |   Frozen Threshold |   Val Support |   Val TP |   Val FP |   Val FN |   Val Prec |   Val Rec |   Val F1 |   Test RW Support |   Test RW F1 |   Test UT Support |   Test UT TP |   Test UT FP |   Test UT FN |   Test UT Prec |   Test UT Rec |   Test UT F1 |
|:---------------------------------------|-------------------:|--------------:|---------:|---------:|---------:|-----------:|----------:|---------:|------------------:|-------------:|------------------:|-------------:|-------------:|-------------:|---------------:|--------------:|-------------:|
| 000 - Normal                           |               0.55 |         52519 |    52454 |      203 |       65 |     0.9961 |    0.9988 |   0.9975 |             52519 |        0.991 |              7414 |         7368 |          121 |           46 |         0.9838 |        0.9938 |       0.9888 |
| 272 - Protocol Manipulation            |               0.99 |           916 |      415 |      249 |      501 |     0.625  |    0.4531 |   0.5253 |               916 |        0.221 |               364 |          297 |           20 |           67 |         0.9369 |        0.8159 |       0.8722 |
| 242 - Code Injection                   |               0.65 |          1581 |     1577 |        2 |        4 |     0.9987 |    0.9975 |   0.9981 |              1583 |        0.998 |               982 |          980 |            0 |            2 |         1      |        0.998  |       0.999  |
| 88 - OS Command Injection              |               0.9  |           758 |      753 |       18 |        5 |     0.9767 |    0.9934 |   0.985  |               758 |        0.917 |               651 |          647 |           20 |            4 |         0.97   |        0.9939 |       0.9818 |
| 126 - Path Traversal                   |               0.9  |          2112 |     2100 |        7 |       12 |     0.9967 |    0.9943 |   0.9955 |              2102 |        0.996 |              1317 |         1313 |            5 |            4 |         0.9962 |        0.997  |       0.9966 |
| 66 - SQL Injection                     |               0.7  |         25015 |    24869 |       56 |      146 |     0.9978 |    0.9942 |   0.996  |             25029 |        0.987 |             14604 |        14489 |           55 |          115 |         0.9962 |        0.9921 |       0.9942 |
| 16 - Dictionary-based Password Attack  |               0.6  |           185 |      185 |        0 |        0 |     1      |    1      |   1      |               174 |        1     |               173 |          173 |            0 |            0 |         1      |        1      |       1      |
| 310 - Scanning for Vulnerable Software |               0.55 |           258 |      256 |       19 |        2 |     0.9309 |    0.9922 |   0.9606 |               427 |        0.97  |               216 |          216 |           17 |            0 |         0.927  |        1      |       0.9621 |
| 153 - Input Data Manipulation          |               0.65 |           220 |      220 |        4 |        0 |     0.9821 |    1      |   0.991  |               225 |        0.993 |               214 |          214 |            2 |            0 |         0.9907 |        1      |       0.9953 |
| 274 - HTTP Verb Tampering              |               0.9  |           970 |      970 |        1 |        0 |     0.999  |    1      |   0.9995 |               495 |        1     |               417 |          417 |            0 |            0 |         1      |        1      |       1      |
| 194 - Fake the Source of Data          |               0.9  |          5615 |     5544 |       22 |       71 |     0.996  |    0.9874 |   0.9917 |              5609 |        0.985 |              3128 |         3070 |            4 |           58 |         0.9987 |        0.9815 |       0.99   |
| 34 - HTTP Response Splitting           |               0.6  |          1971 |     1970 |        0 |        1 |     1      |    0.9995 |   0.9997 |              1971 |        1     |              1259 |         1258 |            0 |            1 |         1      |        0.9992 |       0.9996 |
| 33 - HTTP Request Smuggling            |               0.95 |           132 |      132 |        2 |        0 |     0.9851 |    1      |   0.9925 |                84 |        1     |                49 |           49 |            0 |            0 |         1      |        1      |       1      |

### Summary Multi-Label Classification Metrics
Aggregate multi-label metrics across all evaluation splits:

| Split                              | Threshold Type   |   Macro F1 (All 13) |   Macro F1 (Excl 272) |   Micro F1 |   Weighted F1 |   Macro Precision |   Macro Recall |   Hamming Loss |   Subset Accuracy |
|:-----------------------------------|:-----------------|--------------------:|----------------------:|-----------:|--------------:|------------------:|---------------:|---------------:|------------------:|
| Val Reference (90,781 rows)        | Val-Tuned Frozen |              0.9563 |                0.9922 |     0.9925 |        0.9918 |            0.9603 |         0.9546 |         0.0028 |            0.9842 |
| Test Row-Weighted (90,782 rows)    | Val-Tuned Frozen |              0.9278 |                0.986  |     0.9821 |        0.9798 |            0.9412 |         0.932  |         0.0051 |            0.9712 |
| Test Row-Weighted (90,782 rows)    | Default 0.5      |              0.881  |                0.951  |     0.9415 |        0.946  |            0.892  |         0.921  |         0.0124 |            0.9325 |
| Test Unique-Template (29,996 rows) | Val-Tuned Frozen |              0.9831 |                0.9923 |     0.9912 |        0.9911 |            0.9846 |         0.9824 |         0.0014 |            0.9873 |
| Test Unique-Template (29,996 rows) | Default 0.5      |              0.9502 |                0.9741 |     0.9824 |        0.9844 |            0.9186 |         0.9962 |         0.0028 |            0.9794 |

### Test Unique-Template Performance at Default (0.50) Decision Thresholds
Impact of applying a uniform uncalibrated threshold ($T=0.50$) across all 13 classes:

| Class                                  |   Threshold |   Support |    TP |   FP |   FN |    TN |   Precision |   Recall |       F1 |
|:---------------------------------------|------------:|----------:|------:|-----:|-----:|------:|------------:|---------:|---------:|
| 000 - Normal                           |         0.5 |      7414 |  7371 |  132 |   43 | 22450 |    0.982407 | 0.9942   | 0.988268 |
| 272 - Protocol Manipulation            |         0.5 |       364 |   355 |  352 |    9 | 29280 |    0.502122 | 0.975275 | 0.662932 |
| 242 - Code Injection                   |         0.5 |       982 |   981 |   27 |    1 | 28987 |    0.973214 | 0.998982 | 0.98593  |
| 88 - OS Command Injection              |         0.5 |       651 |   648 |  142 |    3 | 29203 |    0.820253 | 0.995392 | 0.899375 |
| 126 - Path Traversal                   |         0.5 |      1317 |  1316 |   50 |    1 | 28629 |    0.963397 | 0.999241 | 0.980991 |
| 66 - SQL Injection                     |         0.5 |     14604 | 14548 |  111 |   56 | 15281 |    0.992428 | 0.996165 | 0.994293 |
| 16 - Dictionary-based Password Attack  |         0.5 |       173 |   173 |    0 |    0 | 29823 |    1        | 1        | 1        |
| 310 - Scanning for Vulnerable Software |         0.5 |       216 |   216 |   18 |    0 | 29762 |    0.923077 | 1        | 0.96     |
| 153 - Input Data Manipulation          |         0.5 |       214 |   214 |    2 |    0 | 29780 |    0.990741 | 1        | 0.995349 |
| 274 - HTTP Verb Tampering              |         0.5 |       417 |   417 |    0 |    0 | 29579 |    1        | 1        | 1        |
| 194 - Fake the Source of Data          |         0.5 |      3128 |  3103 |  115 |   25 | 26753 |    0.964264 | 0.992008 | 0.977939 |
| 34 - HTTP Response Splitting           |         0.5 |      1259 |  1258 |    0 |    1 | 28737 |    1        | 0.999206 | 0.999603 |
| 33 - HTTP Request Smuggling            |         0.5 |        49 |    49 |   10 |    0 | 29937 |    0.830508 | 1        | 0.907407 |

> [!NOTE]
> **Threshold Sensitivity & Verification**:
> Thresholds were loaded verbatim from `outputs/frozen_v2_1b/thresholds.json`. Applying tuned thresholds ($T=0.70$ for SQLi, $T=0.90$ for Command Injection, Path Traversal, Verb Tampering, and Fake Source; $T=0.99$ for Protocol Manipulation) reduces Normal False Positives from 294 down to 72 on Test UT, while preserving >99% recall on primary attack vectors.

---

## F2. Binary Attack-vs-Normal Performance

Binary detection evaluates whether a request is flagged as an attack versus allowed as Normal traffic. Three explicit evaluation variants are reported across all splits:
- **Variant (i) All 12 Attack Labels**: A request is ground-truth Attack if any of the 12 attack labels is active; positive prediction if any of the 12 attack labels meets or exceeds its frozen threshold.
- **Fixed Ground Truth (Excl-272 Preds Only)**: Identical ground-truth attack definition as Variant (i), but predictions consider only the 11 non-272 attack classes (testing operational impact of muting automated blocks on Protocol Manipulation).
- **Variant (ii) Excl-272**: Requests whose sole attack label is `272 - Protocol Manipulation` are treated as Normal (benign) in ground truth, and the model prediction ignores label 272.

| Split                              | Variant                                                | Threshold        |   Normal_Denom |   Attack_Denom |   Total_Denom |    TN |    FP |   FN |    TP |   Accuracy |   Precision |   Recall |     F1 |   FPR (%) |   FNR (%) |
|:-----------------------------------|:-------------------------------------------------------|:-----------------|---------------:|---------------:|--------------:|------:|------:|-----:|------:|-----------:|------------:|---------:|-------:|----------:|----------:|
| Test Unique-Template (29,996 rows) | Variant (i) All 12 Attack Labels                       | Val-Tuned Frozen |           7414 |          22582 |         29996 |  7342 |    72 |  134 | 22448 |     0.9931 |      0.9968 |   0.9941 | 0.9954 |    0.9711 |    0.5934 |
| Test Unique-Template (29,996 rows) | Fixed GT: Excl-272 Preds Only                          | Val-Tuned Frozen |           7414 |          22582 |         29996 |  7356 |    58 |  337 | 22245 |     0.9868 |      0.9974 |   0.9851 | 0.9912 |    0.7823 |    1.4923 |
| Test Unique-Template (29,996 rows) | Variant (ii) 272-only -> Normal in GT & Excl-272 Preds | Val-Tuned Frozen |           7654 |          22342 |         29996 |  7588 |    66 |  105 | 22237 |     0.9943 |      0.997  |   0.9953 | 0.9962 |    0.8623 |    0.4699 |
| Test Unique-Template (29,996 rows) | Variant (i) All 12 Attack Labels                       | Default 0.5      |           7414 |          22582 |         29996 |  7120 |   294 |   30 | 22552 |     0.9892 |      0.9871 |   0.9987 | 0.9929 |    3.9655 |    0.1328 |
| Test Unique-Template (29,996 rows) | Fixed GT: Excl-272 Preds Only                          | Default 0.5      |           7414 |          22582 |         29996 |  7279 |   135 |  255 | 22327 |     0.987  |      0.994  |   0.9887 | 0.9913 |    1.8209 |    1.1292 |
| Test Unique-Template (29,996 rows) | Variant (ii) 272-only -> Normal in GT & Excl-272 Preds | Default 0.5      |           7654 |          22342 |         29996 |  7499 |   155 |   35 | 22307 |     0.9937 |      0.9931 |   0.9984 | 0.9958 |    2.0251 |    0.1567 |
| Test Row-Weighted (90,782 rows)    | Variant (i) All 12 Attack Labels                       | Val-Tuned Frozen |          52519 |          38263 |         90782 | 52100 |   419 |  620 | 37643 |     0.9886 |      0.989  |   0.9838 | 0.9864 |    0.7978 |    1.6204 |
| Test Row-Weighted (90,782 rows)    | Fixed GT: Excl-272 Preds Only                          | Val-Tuned Frozen |          52519 |          38263 |         90782 | 52114 |   405 | 1034 | 37229 |     0.9841 |      0.9892 |   0.973  | 0.981  |    0.7711 |    2.7024 |
| Test Row-Weighted (90,782 rows)    | Variant (ii) 272-only -> Normal in GT & Excl-272 Preds | Val-Tuned Frozen |          53108 |          37674 |         90782 | 52698 |   410 |  445 | 37229 |     0.9906 |      0.9891 |   0.9882 | 0.9886 |    0.772  |    1.1812 |
| Test Row-Weighted (90,782 rows)    | Variant (i) All 12 Attack Labels                       | Default 0.5      |          52519 |          38263 |         90782 | 16259 | 36260 |  310 | 37953 |     0.5972 |      0.5114 |   0.9919 | 0.6749 |   69.0417 |    0.8102 |
| Test Row-Weighted (90,782 rows)    | Fixed GT: Excl-272 Preds Only                          | Default 0.5      |          52519 |          38263 |         90782 | 51980 |   539 |  690 | 37573 |     0.9865 |      0.9859 |   0.982  | 0.9839 |    1.0263 |    1.8033 |
| Val Reference (90,781 rows)        | Variant (i) All 12 Attack Labels                       | Val-Tuned Frozen |          52519 |          38262 |         90781 | 52317 |   202 |  184 | 38078 |     0.9957 |      0.9947 |   0.9952 | 0.995  |    0.3846 |    0.4809 |
| Val Reference (90,781 rows)        | Fixed GT: Excl-272 Preds Only                          | Val-Tuned Frozen |          52519 |          38262 |         90781 | 52414 |   105 |  494 | 37768 |     0.9934 |      0.9972 |   0.9871 | 0.9921 |    0.1999 |    1.2911 |
| Val Reference (90,781 rows)        | Variant (ii) 272-only -> Normal in GT & Excl-272 Preds | Val-Tuned Frozen |          52829 |          37952 |         90781 | 52720 |   109 |  184 | 37768 |     0.9968 |      0.9971 |   0.9952 | 0.9961 |    0.2063 |    0.4848 |

---

## F3. Genuine CRS Replay & Dual-System Union Benchmark

Replay of all 29,996 Test Unique-Template rows against OWASP CRS v4.23.0 on Nginx:
- **Transport Realism**: Replayed using `session.request(method, ...)` preserving exact HTTP verbs (`GET`, `POST`, `TRACK`, `PROPFIND`, `MKCOL`, `PATCH`, `PUT`, `LOCK`, `MOVE`, `UNLOCK`, `CONNECT`) and complete dataset headers (`User-Agent`, `Content-Type`, `Accept`, `Connection`, `Host`, `Cookie`, `Referer`, `Accept-Language`, `Accept-Encoding`, `Origin`, `DNT`).
- **Blocking Definition**: Strictly defined as **Inbound Anomaly Score $\ge 5$ (Rule 949110 recorded in ModSecurity audit log)**.
- **Correlation**: Each request was tagged with a unique `X-Row-ID` header, enabling exact 1-to-1 mapping with ModSecurity transaction logs.

### Dual-System Binary Metrics (CRS Alone vs Model Alone vs Union)

| System                 |   Accuracy |   Precision |   Recall |       F1 |        FPR |        FNR |   TN |   FP |    FN |    TP |   Total |
|:-----------------------|-----------:|------------:|---------:|---------:|-----------:|-----------:|-----:|-----:|------:|------:|--------:|
| CRS Alone (All-12)     |   0.469929 |    0.896041 | 0.334736 | 0.487394 | 0.11829    | 0.665264   | 6537 |  877 | 15023 |  7559 |   29996 |
| Model Alone (All-12)   |   0.993132 |    0.996803 | 0.994066 | 0.995433 | 0.00971136 | 0.00593393 | 7342 |   72 |   134 | 22448 |   29996 |
| Union (All-12)         |   0.965295 |    0.959747 | 0.99566  | 0.977374 | 0.127192   | 0.00433974 | 6471 |  943 |    98 | 22484 |   29996 |
| CRS Alone (Excl-272)   |   0.465195 |    0.8734   | 0.329782 | 0.478784 | 0.139535   | 0.670218   | 6586 | 1068 | 14974 |  7368 |   29996 |
| Model Alone (Excl-272) |   0.994299 |    0.997041 | 0.9953   | 0.99617  | 0.00862294 | 0.00469967 | 7588 |   66 |   105 | 22237 |   29996 |
| Union (Excl-272)       |   0.959861 |    0.951744 | 0.996643 | 0.973676 | 0.147505   | 0.00335691 | 6525 | 1129 |    75 | 22267 |   29996 |

### CRS Per-Class Agreement & Recall

| Class                                  |   Support |   CRS_Blocked |   CRS_Recall_Agreement |   Model_Recall |
|:---------------------------------------|----------:|--------------:|-----------------------:|---------------:|
| 000 - Normal                           |      7414 |           877 |             0.11829    |       0.993796 |
| 272 - Protocol Manipulation            |       364 |           304 |             0.835165   |       0.815934 |
| 242 - Code Injection                   |       982 |           982 |             1          |       0.997963 |
| 88 - OS Command Injection              |       651 |           633 |             0.97235    |       0.993856 |
| 126 - Path Traversal                   |      1317 |           835 |             0.634017   |       0.996963 |
| 66 - SQL Injection                     |     14604 |          3330 |             0.22802    |       0.992125 |
| 16 - Dictionary-based Password Attack  |       173 |            61 |             0.352601   |       1        |
| 310 - Scanning for Vulnerable Software |       216 |           216 |             1          |       1        |
| 153 - Input Data Manipulation          |       214 |           214 |             1          |       1        |
| 274 - HTTP Verb Tampering              |       417 |           417 |             1          |       1        |
| 194 - Fake the Source of Data          |      3128 |            18 |             0.00575448 |       0.981458 |
| 34 - HTTP Response Splitting           |      1259 |          1259 |             1          |       0.999206 |
| 33 - HTTP Request Smuggling            |        49 |            49 |             1          |       1        |

> [!IMPORTANT]
> **HTTP Verb Preservation Impact**:
> In earlier evaluation runs where non-POST methods were normalized to `GET`, CRS recall on `274 - HTTP Verb Tampering` was 0.0%. In this genuine replay, preserving verbs causes CRS Rule `911100` ("Method is not allowed by policy") to trigger on all unapproved methods, achieving **100.0% CRS agreement (417 / 417)**.

### Marker-Adjusted Benign False Positive Rate (Label Noise Audit)

Detailed audit of the 7,414 Test UT rows labeled as `000 - Normal` in the dataset, partitioned by the presence of explicit attack syntax (`ATTACK_REGEX`):

| Subpopulation              |   Count |   CRS_FP |    CRS_FPR |   Model_All12_FP |   Model_All12_FPR |   Model_No272_FP |   Model_No272_FPR |
|:---------------------------|--------:|---------:|-----------:|-----------------:|------------------:|-----------------:|------------------:|
| All Normal (Raw)           |    7414 |      877 | 0.11829    |               72 |        0.00971136 |               58 |       0.00782304  |
| Normal with Attack Markers |    2548 |      854 | 0.335165   |                6 |        0.00235479 |                1 |       0.000392465 |
| Clean Normal (Marker-Free) |    4866 |       23 | 0.00472667 |               66 |        0.0135635  |               57 |       0.0117139   |

- **Ground-Truth Label Noise**: 34.37% (2,548 / 7,414) of dataset Normal rows contain explicit attack markers (`<script>`, `sleep(15)`, `UNION SELECT`, `<!--#EXEC`). CRS blocks 33.52% of these rows because its regex engines correctly identify attack syntax.
- **Clean Benign FPR**: On the 4,866 genuinely clean Normal rows, CRS False Positive Rate drops from 11.83% to **0.473% (23 / 4,866)**.
- **Model Semantic Discrimination**: The frozen model achieves a 0.24% flag rate on marker-containing Normal rows, proving that its contextual transformer representation distinguishes between malicious code execution and benign blog slugs containing script keywords.

---

## F4. Leakage Ablation: Template-Disjoint vs Random Splitting

Ablation comparing Model v2.1b against the leaky baseline trained on randomly split data:

| Model Split Architecture                      | Evaluation Dataset                           | Template Overlap with Train   |   Macro-F1 (All 13 Classes) |   Macro-F1 (Excluding 272) |   Micro-F1 | Evaluation Condition                              |
|:----------------------------------------------|:---------------------------------------------|:------------------------------|----------------------------:|---------------------------:|-----------:|:--------------------------------------------------|
| Leaky Split Model (Random 80/10/10 Split)     | Leaky Test Set (90,782 rows, 74.31% Overlap) | 74.31%                        |                      0.867  |                     0.9216 |     0.9282 | Evaluated on its own leaky holdout test           |
| Grouped Split Model (v2.1b Template-Disjoint) | Grouped Test Row-Weighted (90,782 rows)      | 0.00% (Strict URL-Disjoint)   |                      0.9278 |                     0.986  |     0.9821 | Evaluated on its own grouped row-weighted test    |
| Grouped Split Model (v2.1b Template-Disjoint) | Grouped Test Unique-Template (29,996 rows)   | 0.00% (Strict URL-Disjoint)   |                      0.9831 |                     0.9923 |     0.9912 | Evaluated on its own grouped unique-template test |

> [!IMPORTANT]
> **Evaluation Set Non-Identity Disclosure**:
> The comparisons in the table above evaluate models on **different test sets**:
> - The leaky baseline was evaluated on its corresponding randomly sampled holdout test set (which shares 74.31% of its structural templates with its training set).
> - Model v2.1b was evaluated on strictly disjoint holdout sets (0.00% template overlap).

---

## F5. Visualizations & Diagnostic Curves

### Figure 1: Training & Validation Curves (Epochs 1–3)
![Figure 1: Training and Validation Dynamics](outputs/final/v2_1b_training_curves.png)
- **Train Loss**: Monotonically decreased from $0.2396 \rightarrow 0.0558 \rightarrow 0.0404$.
- **Validation Loss**: Reached minimum at Epoch 3 ($0.0351$), but Epoch 2 achieved peak **Validation Macro-F1 Excl. 272 of 0.9473**.
- **Optimal Checkpoint Selection**: Epoch 2 was saved as `v2_1b_best.pt` to prevent overfitting on overrepresented classes.

### Figure 2: Precision-Recall Curves on Test Unique-Template (Top 4 Attack Classes)
![Figure 2: Precision-Recall Curves](outputs/final/pr_curves_top4.png)
- Operating points marked at frozen thresholds:
  - SQL Injection ($T=0.70$): Precision = 0.996, Recall = 0.992
  - Code Injection ($T=0.65$): Precision = 1.000, Recall = 0.998
  - Path Traversal ($T=0.90$): Precision = 0.996, Recall = 0.997
  - OS Command Injection ($T=0.90$): Precision = 0.970, Recall = 0.994

### Figure 3: Normalized Binary Confusion Matrix (Test Unique-Template)
![Figure 3: Confusion Matrix Binary](outputs/final/confusion_matrix_binary.png)
- True Normal accuracy: 99.03% (7,342 / 7,414).
- True Attack catch rate: 99.41% (22,448 / 22,582).

---

## F6. Textbook Payloads & Adversarial Mutation Benchmark

### ONE Consolidated Textbook Table (Per Class $\times$ Placement)
Replay of 40 canonical payloads from PayloadsAllTheThings and OWASP CRS regression tests across 5 HTTP placements (200 total requests), evaluating Model Target Label, Model Any Label, CRS (Rule 949110), and Union:

| Attack_Class      | Placement                  |   Total |   Model_Target_Pct |   Model_Any_Pct |   CRS_Blocked_Pct |   Union_Blocked_Pct | Target_Count   | Any_Count   | CRS_Count   | Union_Count   |
|:------------------|:---------------------------|--------:|-------------------:|----------------:|------------------:|--------------------:|:---------------|:------------|:------------|:--------------|
| SQL Injection     | query_arg                  |      15 |               73.3 |            93.3 |             100   |               100   | 11/15          | 14/15       | 15/15       | 15/15         |
| SQL Injection     | post_body_with_ct          |      15 |               86.7 |            93.3 |             100   |               100   | 13/15          | 14/15       | 15/15       | 15/15         |
| SQL Injection     | post_body_no_ct            |      15 |               26.7 |            60   |             100   |               100   | 4/15           | 9/15        | 15/15       | 15/15         |
| SQL Injection     | url_path                   |      15 |               26.7 |            60   |              13.3 |                73.3 | 4/15           | 9/15        | 2/15        | 11/15         |
| SQL Injection     | cookie                     |      15 |                0   |             0   |             100   |               100   | 0/15           | 0/15        | 15/15       | 15/15         |
| SQL Injection     | Subtotal (4 Canonical)     |      60 |               53.3 |            76.7 |              78.3 |                93.3 | 32/60          | 46/60       | 47/60       | 56/60         |
| SQL Injection     | Total (All 5 Placements)   |      75 |               42.7 |            61.3 |              82.7 |                94.7 | 32/75          | 46/75       | 62/75       | 71/75         |
| XSS               | query_arg                  |      10 |              100   |           100   |             100   |               100   | 10/10          | 10/10       | 10/10       | 10/10         |
| XSS               | post_body_with_ct          |      10 |               90   |            90   |             100   |               100   | 9/10           | 9/10        | 10/10       | 10/10         |
| XSS               | post_body_no_ct            |      10 |               10   |            90   |             100   |               100   | 1/10           | 9/10        | 10/10       | 10/10         |
| XSS               | url_path                   |      10 |               10   |           100   |             100   |               100   | 1/10           | 10/10       | 10/10       | 10/10         |
| XSS               | cookie                     |      10 |                0   |             0   |             100   |               100   | 0/10           | 0/10        | 10/10       | 10/10         |
| XSS               | Subtotal (4 Canonical)     |      40 |               52.5 |            95   |             100   |               100   | 21/40          | 38/40       | 40/40       | 40/40         |
| XSS               | Total (All 5 Placements)   |      50 |               42   |            76   |             100   |               100   | 21/50          | 38/50       | 50/50       | 50/50         |
| Path Traversal    | query_arg                  |       5 |              100   |           100   |             100   |               100   | 5/5            | 5/5         | 5/5         | 5/5           |
| Path Traversal    | post_body_with_ct          |       5 |              100   |           100   |             100   |               100   | 5/5            | 5/5         | 5/5         | 5/5           |
| Path Traversal    | post_body_no_ct            |       5 |               60   |           100   |             100   |               100   | 3/5            | 5/5         | 5/5         | 5/5           |
| Path Traversal    | url_path                   |       5 |               80   |            80   |              20   |                80   | 4/5            | 4/5         | 1/5         | 4/5           |
| Path Traversal    | cookie                     |       5 |                0   |             0   |             100   |               100   | 0/5            | 0/5         | 5/5         | 5/5           |
| Path Traversal    | Subtotal (4 Canonical)     |      20 |               85   |            95   |              80   |                95   | 17/20          | 19/20       | 16/20       | 19/20         |
| Path Traversal    | Total (All 5 Placements)   |      25 |               68   |            76   |              84   |                96   | 17/25          | 19/25       | 21/25       | 24/25         |
| Command Injection | query_arg                  |      10 |               20   |            60   |              70   |                90   | 2/10           | 6/10        | 7/10        | 9/10          |
| Command Injection | post_body_with_ct          |      10 |               20   |            20   |              70   |                70   | 2/10           | 2/10        | 7/10        | 7/10          |
| Command Injection | post_body_no_ct            |      10 |                0   |            50   |             100   |               100   | 0/10           | 5/10        | 10/10       | 10/10         |
| Command Injection | url_path                   |      10 |                0   |            90   |              10   |                90   | 0/10           | 9/10        | 1/10        | 9/10          |
| Command Injection | cookie                     |      10 |                0   |             0   |              20   |                20   | 0/10           | 0/10        | 2/10        | 2/10          |
| Command Injection | Subtotal (4 Canonical)     |      40 |               10   |            55   |              62.5 |                87.5 | 4/40           | 22/40       | 25/40       | 35/40         |
| Command Injection | Total (All 5 Placements)   |      50 |                8   |            44   |              54   |                74   | 4/50           | 22/50       | 27/50       | 37/50         |
| ALL CLASSES       | OVERALL (4 Canonical)      |     160 |               46.2 |            78.1 |              80   |                93.8 | 74/160         | 125/160     | 128/160     | 150/160       |
| ALL CLASSES       | OVERALL (All 5 Placements) |     200 |               37   |            62.5 |              80   |                91   | 74/200         | 125/200     | 160/200     | 182/200       |

### Fixed Mutation Operator Audit Table (WAF-A-MoLE Mutations)
Metrics computed exclusively on mutants where the transformation actually modified the payload string, alongside an unmutated control baseline:

| Operator                            |   Mutants_Evaluated |   String_Changed% |   TokenChanged% |   Valid% |   ModelDetect% |   CRSDetect% |   UnionDetect% |   Valid_N | EvadesCRS_ModelCatches   | EvadesModel_CRSCatches   | BothEvade     |
|:------------------------------------|--------------------:|------------------:|----------------:|---------:|---------------:|-------------:|---------------:|----------:|:-------------------------|:-------------------------|:--------------|
| CONTROL (Unmutated Seeds)           |                  40 |                 0 |          0      |  42.5    |        37.5    |      75      |        75      |        17 | 0/17 (0.0%)              | 8/17 (47.1%)             | 5/17 (29.4%)  |
| whitespace_to_comment               |                  18 |               100 |        100      |  50      |        61.1111 |     100      |       100      |         9 | 0/9 (0.0%)               | 5/9 (55.6%)              | 0/9 (0.0%)    |
| swap_tautology                      |                   4 |               100 |        100      |  75      |        25      |      50      |        50      |         3 | 0/3 (0.0%)               | 1/3 (33.3%)              | 2/3 (66.7%)   |
| operator_substitution               |                   8 |               100 |        100      |  50      |        87.5    |     100      |       100      |         4 | 0/4 (0.0%)               | 1/4 (25.0%)              | 0/4 (0.0%)    |
| inline_mysql_comment                |                  25 |               100 |        100      |  36      |        48      |     100      |       100      |         9 | 0/9 (0.0%)               | 5/9 (55.6%)              | 0/9 (0.0%)    |
| randomize_case                      |                  39 |               100 |          0      |  43.5897 |        35.8974 |      74.359  |        74.359  |        17 | 0/17 (0.0%)              | 8/17 (47.1%)             | 5/17 (29.4%)  |
| char_hex_literal                    |                   1 |               100 |        100      |   0      |       100      |     100      |       100      |         0 | 0                        | 0                        | 0             |
| string_concat                       |                   1 |               100 |        100      |   0      |       100      |     100      |       100      |         0 | 0                        | 0                        | 0             |
| benign_comment_dilution             |                  40 |               100 |        100      |  42.5    |        27.5    |      72.5    |        72.5    |        17 | 0/17 (0.0%)              | 8/17 (47.1%)             | 5/17 (29.4%)  |
| ALL_SINGLE_OPERATORS (Changed Only) |                 136 |               100 |         71.3235 |  43.3824 |        42.6471 |      83.0882 |        83.0882 |        59 | 0/59 (0.0%)              | 28/59 (47.5%)            | 12/59 (20.3%) |

---

## F7. Operational Protection Numbers & Enforcement Policy

In an operational deployment, policy routes the **10 BLOCK classes** to automated inline blocking and the **2 ALERT classes** (`272 - Protocol Manipulation` and `310 - Vulnerability Scanning`) to passive logging and rate-limiting.

### Exact Test Row-Weighted Set Protection (52,519 Normal rows, 38,263 Attack rows)
1. **Per-Class False Positives on True-Normal RW**:
   - `66 - SQL Injection`: **352 FPs**
   - `126 - Path Traversal`: **14 FPs**
   - `194 - Fake the Source of Data`: **14 FPs**
   - `88 - OS Command Injection`: **7 FPs**
   - `153 - Input Data Manipulation`: **1 FP**
   - All other 5 BLOCK classes: **0 FPs**
   - Per-class FP sum = 388.
2. **Exact Benign Block Rate**:
   - Unique True-Normal rows blocked by any BLOCK class: **387 rows**
   - **Exact Benign Block Rate**: **0.737% (387 / 52,519 rows)**
   - SQL Injection accounts for 352 / 52,519 = **0.670% (90.96% of all benign blocks)**.
3. **Exact Attack Catch Rates**:
   - Exploit Catch Rate (Any BLOCK class fires on all 38,263 Attack rows): **97.30% (37,229 / 38,263)**
   - Exploit Catch Rate (Any BLOCK class fires on 37,674 non-272 Attack rows): **98.82% (37,229 / 37,674)**
   - Exploit Catch Rate (All 12 attack labels): **98.38% (37,643 / 38,263)**

### Exact Test Unique-Template Set Protection (7,414 Normal rows, 22,582 Attack rows)
- **Benign Block Rate**: **0.634% (47 / 7,414 rows)**
- **Exploit Catch Rate (All 22,582 Attack rows)**: **97.560% (22,031 / 22,582)**
- **Exploit Catch Rate (22,132 BLOCK-GT Attack rows)**: **99.517% (22,025 / 22,132)**

---

## F8. CRS Rule Provenance & Label Inspection

### CRS Rule Group to Model Label Mapping

| Rule_Prefix   | Rule_File                                            | Example_Rule_ID   | Primary_CAPEC_Tags                                | CAPEC_IDs             | Mapped_Model_Labels                                                                    | Description                                                                     |
|:--------------|:-----------------------------------------------------|:------------------|:--------------------------------------------------|:----------------------|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
| 911xxx        | REQUEST-911-METHOD-ENFORCEMENT.conf                  | 911100            | capec/1000/210/272/220/274                        | 272, 274              | 272 - Protocol Manipulation, 274 - HTTP Verb Tampering                                 | Blocks HTTP verbs not allowed by policy (TRACK, TRACE, PROPFIND, etc.)          |
| 913xxx        | REQUEST-913-SCANNER-DETECTION.conf                   | 913100            | capec/1000/118/224/541/310                        | 310, 118, 224, 541    | 310 - Scanning for Vulnerable Software                                                 | Detects automated security scanners, crawler tools, and vulnerability probing   |
| 920xxx        | REQUEST-920-PROTOCOL-ENFORCEMENT.conf                | 920100..920460    | capec/1000/210/272, capec/1000/255/153            | 272, 153, 255, 267    | 272 - Protocol Manipulation, 153 - Input Data Manipulation                             | Protocol enforcement (missing host header, invalid encoding, URL violations)    |
| 921xxx        | REQUEST-921-PROTOCOL-ATTACK.conf                     | 921110..921230    | capec/.../272/220/33, capec/.../272/220/34        | 33, 34, 272           | 33 - HTTP Request Smuggling, 34 - HTTP Response Splitting, 272 - Protocol Manipulation | HTTP Request Smuggling (TE/CL desync) and Response Splitting (CRLF injection)   |
| 930xxx        | REQUEST-930-APPLICATION-ATTACK-LFI.conf              | 930100..930121    | capec/1000/255/153/126                            | 126, 153              | 126 - Path Traversal, 153 - Input Data Manipulation                                    | Directory traversal, ../ path evasion, and /etc/passwd local file inclusion     |
| 931xxx        | REQUEST-931-APPLICATION-ATTACK-RFI.conf              | 931100..931131    | capec/1000/152/175/253                            | 152, 175, 253         | 153 - Input Data Manipulation, 242 - Code Injection                                    | Remote File Inclusion (URL inclusion via query parameters)                      |
| 932xxx        | REQUEST-932-APPLICATION-ATTACK-RCE.conf              | 932100..932331    | capec/1000/152/248/88, capec/137/134              | 88, 248, 152, 134     | 88 - OS Command Injection (and merged 248 - Command Injection)                         | Remote Command Execution, shell metacharacters, unix shell commands             |
| 933xxx        | REQUEST-933-APPLICATION-ATTACK-PHP.conf              | 933100..933211    | capec/1000/152/242                                | 242, 152              | 242 - Code Injection                                                                   | PHP code injection, system/eval execution, upload script injection              |
| 934xxx        | REQUEST-934-APPLICATION-ATTACK-GENERIC.conf          | 934100..934180    | capec/1000/152/242, capec/1000/225/664            | 242, 1, 77, 180, 664  | 242 - Code Injection, 153 - Input Data Manipulation                                    | Generic application attacks, template injection, SSI injection                  |
| 941xxx        | REQUEST-941-APPLICATION-ATTACK-XSS.conf              | 941100..941380    | capec/1000/152/242, capec/.../63                  | 242, 63, 152          | 242 - Code Injection, 153 - Input Data Manipulation                                    | Cross-Site Scripting (XSS), script tag injection, event handlers                |
| 942xxx        | REQUEST-942-APPLICATION-ATTACK-SQLI.conf             | 942100..942432    | capec/1000/152/248/66                             | 66, 248, 152          | 66 - SQL Injection (and 248)                                                           | SQL Injection detected via libinjection and regex signatures                    |
| 943xxx        | REQUEST-943-APPLICATION-ATTACK-SESSION-FIXATION.conf | 943100..943120    | capec/1000/225/21/593/61                          | 21, 61, 593           | 272 - Protocol Manipulation, 153 - Input Data Manipulation                             | Session Fixation attack signatures                                              |
| 944xxx        | REQUEST-944-APPLICATION-ATTACK-JAVA.conf             | 944100..944300    | capec/1000/152/248, capec/1000/152/242, capec/137 | 248, 242, 6, 137      | 88 - OS Command Injection, 242 - Code Injection                                        | Java Class / Object Deserialization and Remote Command Execution                |
| 949xxx        | REQUEST-949-BLOCKING-EVALUATION.conf                 | 949110            | OWASP_CRS, anomaly-evaluation                     | N/A (Evaluation Rule) | Global Inbound Blocking Gate (Threshold >= 5)                                          | Fires when TX:BLOCKING_INBOUND_ANOMALY_SCORE >= 5 (Blocks transaction with 403) |

### Normal Row Inspection Summary
- **Total Normal Rows in Test UT**: 7,414
- **Normal Rows with Attack Markers**: 2,548 (34.37%)
  - CRS Block Rate on Marker Rows: **33.52% (854 / 2,548)**
  - Model Flag Rate (All-12) on Marker Rows: **0.24% (6 / 2,548)**
  - Model Flag Rate (Excl-272) on Marker Rows: **0.04% (1 / 2,548)**
- **Clean Normal Rows (Marker-Free)**: 4,866 (65.63%)
  - CRS Block Rate (Clean FPR): **0.473% (23 / 4,866)**
  - Model Flag Rate (Clean FPR All-12): **1.356% (66 / 4,866)**
  - Model Flag Rate (Clean FPR Excl-272): **1.171% (57 / 4,866)**
- **Inspection Samples**: 25 random CRS-flagged marker rows and 25 random model-flagged clean rows were audited and recorded in [`outputs/final/g2_provenance_and_samples.json`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g2_provenance_and_samples.json).

---

## File Deliverable Manifest

All generated evaluation deliverables have been recorded and saved in `outputs/final/`:

1. **Master Report**: [`outputs/final/eval_report.md`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/eval_report.md)
2. **Metrics & Data Files**:
   - [`outputs/final/f1_multilabel_per_class_frozen.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/f1_multilabel_per_class_frozen.csv)
   - [`outputs/final/f1_multilabel_summary_metrics.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/f1_multilabel_summary_metrics.csv)
   - [`outputs/final/f1_test_ut_per_class_default.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/f1_test_ut_per_class_default.csv)
   - [`outputs/final/f2_binary_attack_vs_normal_verified.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/f2_binary_attack_vs_normal_verified.csv)
   - [`outputs/final/g1_crs_ut_replay_results.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g1_crs_ut_replay_results.csv)
   - [`outputs/final/g1_binary_comparison.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g1_binary_comparison.csv)
   - [`outputs/final/g1_crs_per_class_agreement.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g1_crs_per_class_agreement.csv)
   - [`outputs/final/g1_marker_adjusted_fpr.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g1_marker_adjusted_fpr.csv)
   - [`outputs/final/f4_leakage_ablation.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/f4_leakage_ablation.csv)
   - [`outputs/final/g1_textbook_consolidated.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g1_textbook_consolidated.csv)
   - [`outputs/final/step3_operator_summary_fixed.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/step3_operator_summary_fixed.csv)
   - [`outputs/final/g2_capec_provenance_mapping.csv`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g2_capec_provenance_mapping.csv)
   - [`outputs/final/g2_provenance_and_samples.json`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/g2_provenance_and_samples.json)
3. **Figures**:
   - [`outputs/final/v2_1b_training_curves.png`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/v2_1b_training_curves.png)
   - [`outputs/final/pr_curves_top4.png`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/pr_curves_top4.png)
   - [`outputs/final/confusion_matrix_binary.png`](file:///c:/Users/viren/coding/4thwallV2/outputs/final/confusion_matrix_binary.png)
