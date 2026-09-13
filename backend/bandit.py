"""Per-user linear contextual bandit (spec §4).

score = w . features ; w += alpha * reward * features  (plus a preference step
when the user picks a different candidate). Weights persist per user.
"""
import json
import numpy as np

FEATURES = ["base", "memory", "visual", "history", "pointing", "brevity", "speech", "judgment", "bias"]
ALPHA = 0.12
EXPLORE = 0.05  # epsilon-ish tie-break noise, off during evaluation

# Priors per support mode: nudges, never a diagnosis (spec §1A).
PRIORS = {
    "default":              dict(base=1.0, memory=0.8, visual=0.6, history=0.4,
                                 pointing=0.5, brevity=0.1, speech=0.6, judgment=0.8, bias=0.0),
    "aphasia_word_finding": dict(visual=0.9, pointing=0.8, brevity=0.3),
    "cognitive_fatigue":    dict(memory=1.0, brevity=0.3),
    "aac":                  dict(memory=1.0, history=0.6),
    "speech_language":      dict(memory=1.0),
}


def initial_weights(support_mode: str = "default") -> np.ndarray:
    w = dict(PRIORS["default"])
    w.update(PRIORS.get(support_mode, {}))
    return np.array([w[f] for f in FEATURES], dtype=np.float64)


def vec(features: dict) -> np.ndarray:
    return np.array([float(features.get(f, 0.0)) for f in FEATURES])


class Policy:
    def __init__(self, weights: np.ndarray, version: int = 1, updates: int = 0):
        self.w = weights
        self.version = version
        self.updates = updates

    def score(self, features: dict) -> float:
        return float(np.dot(self.w, vec(features)))

    def update(self, features: dict, reward: float, alpha: float = ALPHA,
               baseline: dict | None = None) -> None:
        """w_new = w_old + alpha * reward * (features - baseline).

        Ranking depends only on feature *differences* between candidates, so the
        update uses the advantage over the candidate-set mean. Without the
        baseline every candidate moves together and the reward adds noise only.
        """
        f = vec(features) - (vec(baseline) if baseline else 0.0)
        self.w = self.w + alpha * reward * f
        self.updates += 1
        self.version += 1

    def prefer(self, chosen: dict, shown: dict, alpha: float = ALPHA) -> None:
        """Supervised preference step: chosen should outrank what we showed."""
        self.w = self.w + alpha * (vec(chosen) - vec(shown))
        self.updates += 1
        self.version += 1

    def as_dict(self) -> dict:
        return dict(zip(FEATURES, [round(x, 4) for x in self.w]))


def load(con, user_id: str, support_mode: str = "default") -> Policy:
    row = con.execute("SELECT * FROM policy WHERE user_id=?", (user_id,)).fetchone()
    if row:
        saved = json.loads(row["weights"])
        fresh = initial_weights(support_mode)
        if isinstance(saved, dict):          # stored by name: new features keep their prior
            w = np.array([saved.get(f, fresh[i]) for i, f in enumerate(FEATURES)], dtype=np.float64)
        else:                                # legacy positional list: only trust an exact match
            w = np.array(saved, dtype=np.float64) if len(saved) == len(FEATURES) else fresh
        return Policy(w, row["version"], row["updates"])
    return Policy(initial_weights(support_mode))


def save(con, user_id: str, p: Policy) -> None:
    con.execute("INSERT INTO policy(user_id,weights,version,updates) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET weights=excluded.weights,"
                "version=excluded.version,updates=excluded.updates",
                (user_id, json.dumps({f: float(x) for f, x in zip(FEATURES, p.w)}),
                 p.version, p.updates))
    con.commit()
