"""Train tabular Q policy and save to models/q_table.npy (run once)."""
from pathlib import Path

import numpy as np

from src.rl_policy import save_policy_json, train_q_table

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    models = root / "models"
    models.mkdir(exist_ok=True)
    q_path = models / "q_table.npy"
    Q = train_q_table(episodes=4000)
    np.save(q_path, Q)
    save_policy_json(Q, models / "q_policy_summary.json")
    print("Saved", q_path)
