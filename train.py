"""Train DeFi Shield anomaly detection models on historical transaction data."""

import argparse
import json
import logging
import os
import random
import time
import hashlib

from models import TransactionData, FeatureExtractor, IsolationForestDetector, DeFiShield


def generate_synthetic_transactions(n: int = 5000, anomaly_ratio: float = 0.05) -> list[TransactionData]:
    """Generate synthetic transaction data for training."""
    txs = []
    n_anomalies = int(n * anomaly_ratio)

    for i in range(n):
        is_anomaly = i < n_anomalies

        if is_anomaly:
            value = random.uniform(50, 1000)
            gas_used = random.randint(100000, 5000000)
            gas_price = random.randint(100, 500)
        else:
            value = random.uniform(0, 50)
            gas_used = random.randint(21000, 500000)
            gas_price = random.randint(10, 100)

        methods = ["0x" + hashlib.md5(str(random.randint(0, 999)).encode()).hexdigest()[:8]
                    for _ in range(20)]

        tx = TransactionData(
            tx_hash="0x" + hashlib.sha256(f"tx_{i}_{time.time()}".encode()).hexdigest(),
            from_address="0x" + hashlib.md5(f"from_{i}".encode()).hexdigest(),
            to_address="0x" + hashlib.md5(f"to_{i}".encode()).hexdigest(),
            value=value,
            gas_used=gas_used,
            gas_price=gas_price,
            method_id=random.choice(methods),
            timestamp=time.time() - random.uniform(0, 86400 * 30),
        )
        txs.append(tx)

    random.shuffle(txs)
    return txs


def train_model(data_path: str = None, output: str = "shield_state.json") -> None:
    """Train the anomaly detection model and save state."""
    if data_path and os.path.exists(data_path):
        logging.info("Loading data from %s", data_path)
        with open(data_path) as f:
            raw = json.load(f)
        txs = [TransactionData(**d) for d in raw]
    else:
        logging.info("Generating synthetic training data...")
        txs = generate_synthetic_transactions(n=5000, anomaly_ratio=0.05)

    shield = DeFiShield()
    shield.train(txs)
    shield.save_state(output)
    logging.info("Model saved to %s", output)

    test_features = shield.extractor.extract(txs[0])
    is_anomaly, score = shield.detector.predict(test_features)
    logging.info("Test prediction: anomaly=%s, score=%.4f", is_anomaly, score)

    n_detected = 0
    for tx in txs[:200]:
        alert = shield.analyze(tx)
        if alert:
            n_detected += 1
    logging.info("Detected %d anomalies in sample of 200 transactions", n_detected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeFi Shield models")
    parser.add_argument("--data", type=str, help="Path to training data JSON")
    parser.add_argument("--output", type=str, default="shield_state.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_model(data_path=args.data, output=args.output)
