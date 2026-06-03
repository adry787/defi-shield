"""DeFi Shield - Anomaly detection models for DeFi transaction monitoring."""

import os
import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TransactionData:
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    method_id: str = ""
    timestamp: float = field(default_factory=time.time)
    block_number: int = 0


class FeatureExtractor:
    """Extract features from raw transaction data for anomaly detection."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.address_tx_counts: dict[str, deque] = {}
        self.method_freq: dict[str, int] = {}

    def extract(self, tx: TransactionData) -> list[float]:
        features = [
            tx.value,
            float(tx.gas_used),
            float(tx.gas_price),
            float(len(tx.from_address)),
            float(len(tx.to_address)),
        ]

        addr_count = self._get_address_frequency(tx.from_address)
        features.append(float(addr_count))

        method_hash = int(tx.method_id[:8], 16) if tx.method_id else 0
        features.append(float(method_hash % 1000))

        value_to_gas_ratio = tx.value / max(tx.gas_used * tx.gas_price / 1e18, 1e-10)
        features.append(value_to_gas_ratio)

        features.append(float(tx.timestamp % 3600))

        return features

    def _get_address_frequency(self, address: str) -> int:
        if address not in self.address_tx_counts:
            self.address_tx_counts[address] = deque(maxlen=self.window_size)
        self.address_tx_counts[address].append(time.time())
        return len(self.address_tx_counts[address])

    def get_feature_dim(self) -> int:
        return 9


class IsolationForestDetector:
    """Lightweight isolation forest for fast anomaly filtering."""

    def __init__(self, n_trees: int = 100, contamination: float = 0.05):
        self.n_trees = n_trees
        self.contamination = contamination
        self.trees: list[dict] = []
        self.threshold: float = 0.0

    def fit(self, data: list[list[float]]) -> None:
        import numpy as np
        arr = np.array(data)
        self.trees = []
        scores = []

        for _ in range(self.n_trees):
            subset_size = min(256, len(arr))
            indices = np.random.choice(len(arr), subset_size, replace=False)
            subset = arr[indices]
            tree = self._build_tree(subset, depth_limit=int(np.log2(subset_size)))
            self.trees.append(tree)

        for row in arr:
            score = self._score(row)
            scores.append(score)

        scores.sort(reverse=True)
        idx = int(len(scores) * self.contamination)
        self.threshold = scores[min(idx, len(scores) - 1)] if scores else 0.5

    def _build_tree(self, data, depth: int = 0, depth_limit: int = 10) -> dict:
        import numpy as np
        if depth >= depth_limit or len(data) <= 1:
            return {"type": "leaf", "size": len(data), "depth": depth}

        n_features = data.shape[1]
        feature_idx = np.random.randint(0, n_features)
        feat_min, feat_max = data[:, feature_idx].min(), data[:, feature_idx].max()

        if feat_min == feat_max:
            return {"type": "leaf", "size": len(data), "depth": depth}

        split_val = np.random.uniform(feat_min, feat_max)
        left_mask = data[:, feature_idx] <= split_val
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feature_idx,
            "threshold": float(split_val),
            "left": self._build_tree(data[left_mask], depth + 1, depth_limit),
            "right": self._build_tree(data[right_mask], depth + 1, depth_limit),
        }

    def _score(self, row: list[float]) -> float:
        path_lengths = []
        for tree in self.trees:
            pl = self._path_length(tree, row)
            path_lengths.append(pl)
        avg_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        c = self._avg_path_length(256)
        return 2 ** (-avg_path / c) if c > 0 else 0.0

    def _path_length(self, node: dict, row: list[float]) -> int:
        if node["type"] == "leaf":
            return node["depth"] + self._avg_path_length(node["size"])
        if row[node["feature"]] <= node["threshold"]:
            return self._path_length(node["left"], row)
        return self._path_length(node["right"], row)

    @staticmethod
    def _avg_path_length(n: int) -> float:
        import math
        if n <= 1:
            return 0
        return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def predict(self, row: list[float]) -> tuple[bool, float]:
        score = self._score(row)
        return score > self.threshold, score


class AnomalyAlert:
    def __init__(self, tx: TransactionData, score: float, reasons: list[str]):
        self.tx = tx
        self.score = score
        self.reasons = reasons
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx.tx_hash,
            "from": self.tx.from_address,
            "to": self.tx.to_address,
            "value": self.tx.value,
            "anomaly_score": self.score,
            "reasons": self.reasons,
            "timestamp": self.timestamp,
        }


class DeFiShield:
    """Main anomaly detection pipeline for DeFi transactions."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.detector = IsolationForestDetector(n_trees=50, contamination=0.03)
        self.alerts: list[AnomalyAlert] = []
        self.is_trained = False

    def train(self, historical_txs: list[TransactionData]) -> None:
        logger.info("Training on %d transactions...", len(historical_txs))
        features = [self.extractor.extract(tx) for tx in historical_txs]
        self.detector.fit(features)
        self.is_trained = True
        logger.info("Training complete. Threshold: %.4f", self.detector.threshold)

    def analyze(self, tx: TransactionData) -> Optional[AnomalyAlert]:
        if not self.is_trained:
            logger.warning("Model not trained. Run train() first.")
            return None

        features = self.extractor.extract(tx)
        is_anomaly, score = self.detector.predict(features)

        if not is_anomaly:
            return None

        reasons = self._generate_reasons(tx, score)
        alert = AnomalyAlert(tx=tx, score=score, reasons=reasons)
        self.alerts.append(alert)
        logger.warning("ANOMALY DETECTED: %s (score=%.4f)", tx.tx_hash[:16], score)
        return alert

    def _generate_reasons(self, tx: TransactionData, score: float) -> list[str]:
        reasons = []
        if tx.value > 100:
            reasons.append("High transaction value")
        if tx.gas_price > 200:
            reasons.append("Abnormally high gas price")
        if score > 0.8:
            reasons.append("Extreme anomaly score")
        freq = self.extractor._get_address_frequency(tx.from_address)
        if freq > 50:
            reasons.append("High-frequency address")
        if not reasons:
            reasons.append("Pattern mismatch detected")
        return reasons

    def get_alerts(self, min_score: float = 0.0) -> list[dict]:
        return [a.to_dict() for a in self.alerts if a.score >= min_score]

    def save_state(self, path: str = "shield_state.json") -> None:
        state = {
            "threshold": self.detector.threshold,
            "is_trained": self.is_trained,
            "alert_count": len(self.alerts),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str = "shield_state.json") -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.detector.threshold = state["threshold"]
        self.is_trained = state["is_trained"]
        return True


def run_monitoring(rpc_url: str = "ws://localhost:8546"):
    shield = DeFiShield()
    if not shield.load_state():
        logger.error("No trained model found. Run train.py first.")
        return

    logger.info("Starting DeFi Shield monitoring on %s", rpc_url)

    while True:
        try:
            sample = TransactionData(
                tx_hash=hashlib.sha256(str(time.time()).encode()).hexdigest(),
                from_address="0x" + "a" * 40,
                to_address="0x" + "b" * 40,
                value=0.0,
                gas_used=21000,
                gas_price=20,
            )
            shield.analyze(sample)
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down DeFi Shield")
            break
        except Exception as e:
            logger.error("Monitor error: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_monitoring()
