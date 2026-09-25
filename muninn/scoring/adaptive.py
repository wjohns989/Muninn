"""
Muninn Adaptive Importance
--------------------------
Self-supervised importance learning. No human feedback is required: the
system labels its own predictions from what it later observes.

Target
    For a memory at time T: "will it be retrieved again, in a session that had
    not already retrieved it, within the next ``horizon`` days?" The outcome is
    read from the access-event log once T + horizon has passed.

Features
    ACT-R base-level activation ``B = ln(sum_j age_j^-d)`` over the memory's
    presentations (creation plus every retrieval), at several decay rates d so
    the learner effectively fits the decay rate from data (Anderson & Schooler,
    1991: need probability tracks this power-law history). Plus retrieval and
    distinct-session counts, age, ingestion-time novelty and provenance.

Learning and self-evaluation
    Online logistic regression with AdaGrad. Every prediction is stored before
    its outcome is known and scored when the outcome arrives, before the model
    learns from it (prequential evaluation). Rolling AUC is compared against the
    legacy hand-weighted importance on the same examples; the learned model is
    used only while it beats legacy by ``margin`` (with hysteresis), so the
    system switches itself over, and back, from its own evidence.
"""

import json
import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from muninn.core.types import MemoryRecord
from muninn.scoring.importance import calculate_provenance_weight

DECAY_RATES = (0.2, 0.5, 0.8)
FEATURE_NAMES = tuple(f"activation_d{d}" for d in DECAY_RATES) + (
    "log_retrievals",
    "log_sessions",
    "log_age_days",
    "novelty",
    "provenance",
    "bias",
)
_MIN_AGE_DAYS = 1.0 / 1440.0  # one minute; ACT-R ages are floored to avoid t^-d blow-up
_SECONDS_PER_DAY = 86400.0


def base_level_activation(ages_days: Iterable[float], decay: float) -> float:
    """ACT-R base-level activation ln(sum age^-d) over presentation ages."""
    total = sum(max(age, _MIN_AGE_DAYS) ** (-decay) for age in ages_days)
    return math.log(total) if total > 0 else math.log(_MIN_AGE_DAYS ** (-decay))


def build_features(
    record: MemoryRecord,
    access_times: Sequence[float],
    session_count: int,
    now: float,
) -> List[float]:
    """Feature vector for ``record`` at ``now`` from accesses that happened before ``now``."""
    presentations = [record.created_at] + [t for t in access_times if t <= now]
    ages = [(now - t) / _SECONDS_PER_DAY for t in presentations]
    age_days = max(0.0, (now - record.created_at) / _SECONDS_PER_DAY)
    return [base_level_activation(ages, d) for d in DECAY_RATES] + [
        math.log1p(len(presentations) - 1),
        math.log1p(max(0, session_count)),
        math.log1p(age_days),
        float(record.novelty_score),
        calculate_provenance_weight(record.provenance),
        1.0,
    ]


def outcome_label(
    events: Sequence[Tuple[float, Optional[str]]], predicted_at: float, horizon_seconds: float
) -> int:
    """1 if retrieved in (T, T+horizon] by a session that had not retrieved it by T."""
    prior_sessions = {sid for ts, sid in events if ts <= predicted_at and sid is not None}
    window_end = predicted_at + horizon_seconds
    for ts, sid in events:
        if predicted_at < ts <= window_end and (sid is None or sid not in prior_sessions):
            return 1
    return 0


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Mann-Whitney AUC with average ranks for ties; None without both classes."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    positive_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


class AdaptiveImportanceModel:
    """Online logistic model of re-retrieval probability with a self-evaluation gate."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
        window: int = 2000,
        min_examples: int = 200,
        min_class_examples: int = 20,
        margin: float = 0.02,
    ):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.min_examples = min_examples
        self.min_class_examples = min_class_examples
        self.margin = margin
        self.weights = [0.0] * len(FEATURE_NAMES)
        self._grad_sq = [1e-8] * len(FEATURE_NAMES)
        self.examples_seen = 0
        self.active = False
        self._window: Deque[Tuple[float, float, int]] = deque(maxlen=window)

    # --- prediction ---

    def predict(self, features: Sequence[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, features))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def base_rate(self) -> float:
        """Smoothed rate of positive outcomes over the evaluation window."""
        positives = sum(y for _, _, y in self._window)
        return (positives + 1.0) / (len(self._window) + 2.0)

    def importance(self, probability: float) -> float:
        """Map re-retrieval probability to [0, 1] relative to the base rate (0.5 = average)."""
        return probability / (probability + self.base_rate)

    # --- learning ---

    def observe(self, features: Sequence[float], p_model: float, legacy_score: float, label: int) -> None:
        """Score a stored prediction against its outcome, then learn from it."""
        self._window.append((p_model, legacy_score, int(label)))
        error = self.predict(features) - label
        for i, x in enumerate(features):
            gradient = error * x + self.l2 * self.weights[i]
            self._grad_sq[i] += gradient * gradient
            self.weights[i] -= self.learning_rate * gradient / math.sqrt(self._grad_sq[i])
        self.examples_seen += 1
        self._update_gate()

    def evaluation(self) -> Dict[str, Any]:
        model_scores = [p for p, _, _ in self._window]
        legacy_scores = [s for _, s, _ in self._window]
        labels = [y for _, _, y in self._window]
        log_loss = None
        if labels:
            eps = 1e-6
            log_loss = -sum(
                y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                for p, y in zip(model_scores, labels)
            ) / len(labels)
        return {
            "window": len(labels),
            "positives": sum(labels),
            "model_auc": rank_auc(model_scores, labels),
            "legacy_auc": rank_auc(legacy_scores, labels),
            "model_log_loss": log_loss,
        }

    def _update_gate(self) -> None:
        stats = self.evaluation()
        positives, window = stats["positives"], stats["window"]
        enough = (
            window >= self.min_examples
            and positives >= self.min_class_examples
            and window - positives >= self.min_class_examples
        )
        model_auc, legacy_auc = stats["model_auc"], stats["legacy_auc"]
        if not enough or model_auc is None or legacy_auc is None:
            return
        if not self.active and model_auc >= legacy_auc + self.margin:
            self.active = True
        elif self.active and model_auc < legacy_auc - self.margin:
            self.active = False

    # --- persistence ---

    def to_json(self) -> str:
        return json.dumps({
            "version": 1,
            "features": list(FEATURE_NAMES),
            "weights": self.weights,
            "grad_sq": self._grad_sq,
            "examples_seen": self.examples_seen,
            "active": self.active,
            "window": list(self._window),
        })

    def load_json(self, payload: Optional[str]) -> None:
        if not payload:
            return
        try:
            state = json.loads(payload)
        except (TypeError, ValueError):
            return
        if state.get("version") != 1 or state.get("features") != list(FEATURE_NAMES):
            return  # feature set changed: start fresh rather than misapply weights
        self.weights = [float(w) for w in state["weights"]]
        self._grad_sq = [float(g) for g in state["grad_sq"]]
        self.examples_seen = int(state.get("examples_seen", 0))
        self.active = bool(state.get("active", False))
        self._window.clear()
        self._window.extend((float(p), float(s), int(y)) for p, s, y in state.get("window", []))

    def status(self) -> Dict[str, Any]:
        stats = self.evaluation()
        return {
            "active": self.active,
            "examples_seen": self.examples_seen,
            "base_rate": round(self.base_rate, 4),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
        }
