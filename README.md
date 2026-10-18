# 4thwall – Transformer-based Hybrid WAF

---

A hybrid Web Application Firewall that combines **ModSecurity** (rule-based) with a **Transformer (DistilBERT)** classifier in a two-layer inline defense pipeline. The ML model is hosted on [HuggingFace Hub](https://huggingface.co/viitheone/waf-model) and downloaded automatically at startup.

## Architecture

```
                    ┌─────────────────┐
                    │     Client      │
                    └────────┬────────┘
                             │
                       port 80
                             │
                             ▼
                    ┌─────────────────┐
                    │     Nginx       │
                    │  + ModSecurity  │     Layer 1: Rule-based detection
                    │  (CRS rules)   │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
             MALICIOUS            BENIGN
                   │                   │
                   ▼                   ▼
            403 Forbidden     ┌─────────────────┐
            + log to          │   FastAPI API    │
            access.log        │  WAFClassifier   │     Layer 2: ML-based detection
                              │  (DistilBERT)    │
                              └────────┬─────────┘
                                       │
                             ┌─────────┴─────────┐
                             │                   │
                       MALICIOUS            BENIGN
                             │                   │
                             ▼                   ▼
                      403 Forbidden       ┌──────────────┐
                      + log to            │     DVWA     │
                      decision_log        │  (backend)   │
                                          └──────────────┘

        ┌─────────────────┐
        │  Background:    │     Tails access.log every 60s,
        │  modsec_block   │     logs ModSec 403s to decision_log
        │  _logger        │     so they appear in the dashboard.
        └────────┬────────┘
                 │
                 ▼
        decision_log.json
        (BLOCK/ALERT/ALLOW + blocked_by: modsec | ml_model)
                 │
                 ▼
        ┌─────────────────┐
        │  Dashboard UI   │     port 3000
        │  (React + Vite) │
        └─────────────────┘
```

### Request Flow

1. **Client → Nginx/ModSecurity (port 80)** — CRS rules evaluate the request.
   - If **malicious** → Nginx returns `403 Forbidden` directly. The background `modsec_block_logger` picks up the 403 from `access.log` and writes it to `decision_log.json` with `blocked_by: "modsec"`.
   - If **benign** → request is forwarded to the API.

2. **API (ML scoring)** — The FastAPI catch-all route scores the request with the DistilBERT WAF classifier.
   - If **malicious** (ML score > 0.9) → returns `403 Forbidden`, logs to `decision_log.json` with `blocked_by: "ml_model"`.
   - If **medium risk** (ML score > 0.6) → logs an `ALERT` but allows the request through.
   - If **benign** → proxies the request to DVWA and returns the response.

3. **Dashboard (port 3000)** — Reads `decision_log.json` and shows live traffic, attack distribution (ModSec vs ML blocks), and AI model status.

## Prerequisites

- Docker and Docker Compose

For local development (training only):
- Python 3.10+

## Quick Start

1. **Clone and enter the project:**
   ```bash
   cd 4thwall
   ```

2. **Copy environment template:**
   ```bash
   copy .env.example .env
   ```
   Edit `.env` if you need different settings.

3. **Build and start all services:**
   ```bash
   docker compose up --build -d
   ```
   On first startup, the API container downloads the model from [HuggingFace Hub](https://huggingface.co/viitheone/waf-model) (~260MB). Subsequent starts use the cached model from a Docker volume.

4. **Access the stack:**

   | Service            | URL                    | Notes                                                       |
   |--------------------|------------------------|-------------------------------------------------------------|
   | **Nginx (WAF)**    | http://localhost:80    | Entry point — all DVWA traffic goes through ModSec + ML.    |
   | **Dashboard UI**   | http://localhost:3000  | Live traffic, attack stats, AI model status.                |

   > **Note:** The API (port 8000) and DVWA (port 80 inside container) are **internal only** — not exposed to the host. All traffic must go through Nginx on port 80.

5. **Useful commands:**
   - View logs: `docker compose logs -f`
   - Stop: `docker compose down`
   - Rebuild API only: `docker compose build api`

## Model

The WAF classifier is a fine-tuned **DistilBERT** model hosted at [`viitheone/waf-model`](https://huggingface.co/viitheone/waf-model) on HuggingFace Hub.

- **Pipeline:** Text Classification (binary: benign vs malicious)
- **Format:** Safetensors
- **Download:** Automatic at API startup via `transformers.AutoModelForSequenceClassification.from_pretrained()`
- **Caching:** Stored in a named Docker volume (`hf_cache`) to persist across container restarts

To use a **local model** instead (e.g. after retraining), set `MODEL_PATH` to a local directory:
```env
MODEL_PATH=./models/waf_model
```

## Training

1. Place your training CSV in `data/` (columns: `method`, `path` or `url`, `query` or `args`, `status`, `user_agent`, `request_time`, `label` with label 0=benign, 1=malicious).

2. Install dependencies locally:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python -m ml.train --data_path data/train.csv
   ```
   Optional: `--output_dir ./models/waf_model` and `--epochs 3`.

4. Model and tokenizer are saved under `MODEL_SAVE_PATH` (default `./models/waf_model`). Best model is chosen by validation F1.

5. To use your retrained model, either:
   - Upload it to HuggingFace Hub and update `MODEL_PATH` in `.env`
   - Or set `MODEL_PATH=./models/waf_model` and mount the models directory in `docker-compose.yml`

## Policy Settings

| Condition                | Action    | Reason                       |
|--------------------------|-----------|------------------------------|
| ModSecurity blocked      | **BLOCK** | ModSecurity rule triggered   |
| ML score > 0.9           | **BLOCK** | High ML risk score           |
| ML score > 0.6           | **ALERT** | Medium ML risk score         |
| Otherwise                | **ALLOW** | Low risk                     |

## Decision Logging

Every request that passes through the system is logged to `logs/decision_log.json` with the following fields:

| Field                 | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `timestamp`           | ISO 8601 timestamp                                             |
| `request_text`        | Serialized request (method, path, query, status, UA, time)     |
| `modsecurity_blocked` | Whether ModSecurity blocked the request                        |
| `ml_score`            | ML model risk score (0.0–1.0)                                  |
| `action`              | Final action: `BLOCK`, `ALERT`, or `ALLOW`                     |
| `reason`              | Human-readable reason for the action                           |
| `blocked_by`          | Which layer blocked: `"modsec"`, `"ml_model"`, or `null`       |

## Troubleshooting

- **503 on /score**: Model failed to download from HuggingFace Hub. Check internet connectivity and that `MODEL_PATH` is correct (default: `viitheone/waf-model`).
- **Import errors**: Run from project root and ensure `requirements.txt` is installed.
- **Empty dashboard**: Traffic must go through **http://localhost:80** (Nginx WAF). ModSec blocks appear in the dashboard with ~60 second delay (background log tailer interval).
- **Port 80 not loading**: Run `docker compose ps` — all services should be `Up`. The CRS nginx image listens on **8080 inside the container**; compose maps **host `80` → container `8080`**.
- **Nginx `Restarting` + bind-mount conf**: If `nginx/conf.d/99-waf-host-access-log.conf` did not exist the first time you ran Compose, Docker may have created a **directory** with that name. Remove that folder, ensure the path is a **regular file**, then `docker compose up -d --force-recreate nginx`.
- **"WAF doesn't block"**: ModSecurity blocks at the Nginx layer (403). The ML model blocks inline in the API for requests that pass ModSec. Test with: `curl "http://localhost:80/?id=1' OR 1=1--"`.
- **Docker build timeout / huge downloads**: The API Dockerfile installs **CPU-only** PyTorch from PyTorch's wheel index. If a build times out, retry with `docker compose build --no-cache api`.
- **Dashboard can't reach API**: The frontend proxies `/api` to the API container over the Docker network. Run `docker compose ps` to confirm both are running.

## Known Issues

- No online learning; model is fixed after training.
- ModSec blocks appear in the dashboard with ~60s delay (access log tailing interval).
- Docker Compose sets `MODSEC_RULE_ENGINE=On`; local `nginx/nginx.conf` samples may still be detection-only if running Nginx manually.
- Policy engine has exactly the specified rules (ModSec block, then ML thresholds 0.9 / 0.6).

