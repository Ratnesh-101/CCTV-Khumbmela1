"""
Wi-Fi drone traffic anomaly detection aligned with ISOT lab methodology:
Isolation Forest on per-flow features (see isot-lab repo feature extraction).

Reference: https://github.com/isot-lab/Drone-Anomaly-Detection-Dataset-and-Unsupervised-Machine-Learning

Optional: MiniLM embeddings of text fingerprints for distance-from-normal (cosine).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Columns from ISOT Feature_extraction.pcap_evaluation (numeric / one-hot); ts + name excluded at runtime.
ISOT_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "Payload_Length",
    "Protocol Type",
    "Duration",
    "Drone_port",
    "OUI",
    "Entropy",
    "Var_Payload",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "urg_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "max_duration",
    "min_duration",
    "sum_duration",
    "average_duration",
    "std_duration",
    "CoAP",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
    "DS status",
    "Fragments",
    "Sequence number",
    "Protocol Version",
    "flow_idle_time",
    "flow_active_time",
)


def _pick_numeric_columns(df: pd.DataFrame) -> List[str]:
    present = [c for c in ISOT_NUMERIC_COLUMNS if c in df.columns]
    if not present:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        drop = {"Label", "label", "ts"}
        present = [c for c in num if c not in drop]
    return present


def dataframe_to_xy(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Split features / optional binary label (1=attack, -1 or 0=regular)."""
    lc = label_col
    if lc is None:
        for cand in ("Label", "label"):
            if cand in df.columns:
                lc = cand
                break
    y = None
    work = df.copy()
    if lc and lc in work.columns:
        y = work[lc].to_numpy()
        work = work.drop(columns=[lc])
    feat_cols = _pick_numeric_columns(work)
    X = work[feat_cols].apply(pd.to_numeric, errors="coerce")
    return X, y


@dataclass
class WifiIsolationResult:
    """Per-row scores and low-dimensional embedding for visualization."""

    anomaly_01: np.ndarray  # 0=normal, 1=unauthorized / anomalous
    iforest_score: np.ndarray  # sklearn score_samples (higher = more normal)
    embedding_2d: np.ndarray  # PCA of scaled features
    feature_columns: List[str]
    meta: Dict[str, Any]


class WifiDroneIsolationModel:
    """
    Train Isolation Forest on normal (authorized) Wi-Fi flow features only — same spirit as ISOT tuning.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: float = 0.75,
        max_features: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._if: Optional[IsolationForest] = None
        self._feat_cols: List[str] = []
        self._train_score_mean: float = 0.0
        self._train_score_std: float = 1.0

    def fit(self, X: pd.DataFrame) -> "WifiDroneIsolationModel":
        cols = list(X.columns)
        self._feat_cols = cols
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        Xi = self._imputer.fit_transform(X)
        Xs = self._scaler.fit_transform(Xi)
        self._if = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._if.fit(Xs)
        ref = self._if.score_samples(Xs)
        self._train_score_mean = float(np.mean(ref))
        self._train_score_std = float(np.std(ref) + 1e-9)
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._imputer is None or self._scaler is None or self._if is None:
            raise RuntimeError("Model not fitted")
        X = X.reindex(columns=self._feat_cols, fill_value=np.nan)
        Xi = self._imputer.transform(X)
        return self._scaler.transform(Xi)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._transform(X)
        return self._if.score_samples(Xs)  # higher = more like training (normal)

    def anomaly_score_01(self, X: pd.DataFrame) -> np.ndarray:
        """Map score_samples to [0,1]; higher = more likely unauthorized / anomalous."""
        s = self.score_samples(X)
        # Lower sklearn score => more anomalous
        z = (self._train_score_mean - s) / (3.0 * self._train_score_std)
        return np.clip(z, 0.0, 1.0)

    def pca_embedding(self, X: pd.DataFrame, n_components: int = 2) -> np.ndarray:
        Xs = self._transform(X)
        pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=self.random_state)
        return pca.fit_transform(Xs)


def run_wifi_pipeline(
    df: pd.DataFrame,
    train_normal_mask: Optional[np.ndarray] = None,
    label_col: Optional[str] = None,
) -> Tuple[WifiDroneIsolationModel, WifiIsolationResult]:
    """
    Fit IF on rows selected as normal (authorized). If mask is None, uses all rows
    (unsupervised baseline). If labels exist (ISOT: -1 regular, 1 attack), fits on regular only.
    """
    X, y = dataframe_to_xy(df, label_col=label_col)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No numeric feature columns found — expected ISOT-style CSV.")

    if train_normal_mask is None and y is not None:
        train_normal_mask = np.isin(y, (-1, 0))

    if train_normal_mask is not None and train_normal_mask.shape[0] == len(X):
        X_fit = X.loc[train_normal_mask].copy()
        if len(X_fit) < 10:
            X_fit = X.copy()
    else:
        X_fit = X.copy()

    model = WifiDroneIsolationModel().fit(X_fit)
    a01 = model.anomaly_score_01(X)
    ss = model.score_samples(X)
    emb = model.pca_embedding(X, n_components=2)
    return model, WifiIsolationResult(
        anomaly_01=a01,
        iforest_score=ss,
        embedding_2d=emb,
        feature_columns=list(X.columns),
        meta={"fit_rows": len(X_fit), "score_rows": len(X)},
    )


def row_fingerprint_text(row: pd.Series) -> str:
    """Compact text line for embedding similarity (ISOT-style fields when present)."""
    parts = []
    for key in (
        "Payload_Length",
        "Entropy",
        "Drone_port",
        "OUI",
        "Var_Payload",
        "Rate",
        "Srate",
        "Drate",
        "DS status",
        "Protocol Version",
    ):
        if key in row.index and pd.notna(row[key]):
            parts.append(f"{key}={row[key]}")
    if not parts:
        parts = [f"{k}={row[k]}" for k in row.index[:12] if pd.notna(row[k])]
    return "wifi_flow " + " ".join(parts)


def embedding_distance_anomaly(
    df: pd.DataFrame,
    normal_mask: np.ndarray,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_rows: int = 256,
) -> Optional[np.ndarray]:
    """
    Embed row fingerprints; distance from mean normal embedding => anomaly proxy [0,1].
    Returns None if sentence_transformers unavailable or empty.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        return None

    X, _ = dataframe_to_xy(df)
    n = min(len(X), max_rows)
    if n == 0:
        return None
    X = X.iloc[:n]
    mask = normal_mask[:n] if len(normal_mask) >= n else np.ones(n, dtype=bool)

    texts = [row_fingerprint_text(X.iloc[i]) for i in range(n)]
    emb = HuggingFaceEmbeddings(model_name=model_name)
    vecs = np.array(emb.embed_documents(texts), dtype=np.float64)
    normal_vecs = vecs[mask[: vecs.shape[0]]]
    if normal_vecs.shape[0] == 0:
        centroid = vecs.mean(axis=0)
    else:
        centroid = normal_vecs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    sim = vecs @ centroid
    # cosine similarity high => like normal; anomaly = 1 - sim mapped
    sim_min, sim_max = float(sim.min()), float(sim.max())
    span = max(sim_max - sim_min, 1e-9)
    out = 1.0 - (sim - sim_min) / span
    full = np.zeros(len(df))
    full[:n] = np.clip(out, 0.0, 1.0)
    return full


def synthetic_demo_frames(n_normal: int = 400, n_anomaly: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic ISOT-like rows: normal tight cluster; anomalies shifted + noise."""
    rng = np.random.default_rng(seed)
    k = len(ISOT_NUMERIC_COLUMNS)
    base = rng.normal(0.0, 1.0, size=(n_normal, k))
    attack = rng.normal(2.5, 1.8, size=(n_anomaly, k))
    X = np.vstack([base, attack])
    df = pd.DataFrame(X, columns=list(ISOT_NUMERIC_COLUMNS))
    df["Label"] = np.array([-1] * n_normal + [1] * n_anomaly)
    return df
