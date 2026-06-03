# DeFi Shield 🛡️

AI-powered anomaly detection for DeFi protocols. Monitors on-chain transactions, identifies suspicious patterns, and alerts on potential exploits in real-time.

## Features

- Real-time transaction monitoring via WebSocket
- Isolation Forest + Autoencoder anomaly detection
- Configurable alert thresholds
- PostgreSQL storage for historical analysis
- REST API for dashboard integration

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # configure RPC_URL and DB connection
python train.py       # train models on historical data
python models.py      # run real-time monitoring
```

## GPU Requirements

| Component | GPU | VRAM | Notes |
|-----------|-----|------|-------|
| Training (Autoencoder) | CUDA-capable | 2GB+ | CPU fallback available |
| Inference | None | — | Runs on CPU |

## Architecture

Transaction data → Feature extraction → Isolation Forest (fast filter) → Autoencoder (deep analysis) → Alert system
