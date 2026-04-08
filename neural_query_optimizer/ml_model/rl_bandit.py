from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class EpsilonGreedyBandit:
    """Lightweight RL extension for online plan selection refinement.

    State is represented by a bucketed query signature and action is a plan_id.
    Reward should be higher for lower latency, e.g., reward = -latency_ms.
    """

    epsilon: float = 0.1
    value_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    count_table: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def choose_action(self, state: str, action_scores: Dict[str, float]) -> str:
        if np.random.random() < self.epsilon or state not in self.value_table:
            return str(np.random.choice(list(action_scores.keys())))

        state_values = self.value_table[state]
        return min(action_scores.keys(), key=lambda a: action_scores[a] - state_values.get(a, 0.0))

    def update(self, state: str, action: str, reward: float) -> None:
        self.value_table.setdefault(state, {})
        self.count_table.setdefault(state, {})

        old_value = self.value_table[state].get(action, 0.0)
        count = self.count_table[state].get(action, 0) + 1
        self.count_table[state][action] = count
        self.value_table[state][action] = old_value + (reward - old_value) / count
