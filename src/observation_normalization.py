"""Observation-only running statistics; inspired by SimBa (Lee et al., ICLR 2025),
Section 4, equations 3-4: https://arxiv.org/abs/2410.09754.
Independently implemented Welford estimator over training decision states.
Only update() mutates statistics. Network forwards never update them.
"""

from dataclasses import dataclass

import numpy as np
import torch

EPSILON = 1e-8
CONVENTION = "welford_population_float32_decision_states_v1"


@dataclass(frozen=True)
class FrozenObservationTransform:
    """Independent cached tensors, with no statistics-update interface."""

    mean: torch.Tensor
    inv_std: torch.Tensor

    def transform(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim not in (1, 2) or obs.shape[-1] != self.mean.numel():
            raise ValueError("Expected observation shape (D,) or (B,D)")
        if obs.dtype != torch.float32 or obs.device != self.mean.device:
            raise ValueError("Observations must be float32 on the normalizer device")
        if not torch.isfinite(obs).all():
            raise ValueError("Nonfinite observation")
        result = (obs - self.mean) * self.inv_std
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite normalized observation")
        return result


class ObservationNormalizer:
    def __init__(self, obs_dim: int, device: torch.device | str = "cpu") -> None:
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, int) or obs_dim < 1:
            raise ValueError("obs_dim must be a positive integer")
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.m2 = np.zeros(obs_dim, dtype=np.float64)
        self._refresh()

    def _refresh(self) -> None:
        variance = self.m2 / self.count if self.count else np.ones(self.obs_dim)
        inv_std = 1.0 / np.sqrt(np.maximum(variance, 0.0) + EPSILON)
        if not self.count:
            inv_std = np.ones(self.obs_dim)  # exact identity before the first update
        self.cached_mean = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
        self.cached_inv_std = torch.tensor(inv_std, dtype=torch.float32, device=self.device)

    def update(self, obs: np.ndarray) -> None:
        """Incorporate ONE raw current observation before its training action.
        Cast to replay precision first; never call on replay samples or evaluation.
        """
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (self.obs_dim,) or not np.isfinite(x).all():
            raise ValueError("Expected one finite observation of shape (D,)")
        x = x.astype(np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self._refresh()

    def transform(self, obs: torch.Tensor) -> torch.Tensor:
        return FrozenObservationTransform(self.cached_mean, self.cached_inv_std).transform(obs)

    def snapshot(self) -> FrozenObservationTransform:
        return FrozenObservationTransform(self.cached_mean.clone(), self.cached_inv_std.clone())

    def specification(self) -> dict:
        return {"version": 1, "convention": CONVENTION, "obs_dim": self.obs_dim,
                "epsilon": EPSILON, "clipping": None}

    def state_dict(self) -> dict:
        return {**self.specification(), "count": self.count,
                "mean": torch.from_numpy(self.mean.copy()), "m2": torch.from_numpy(self.m2.copy())}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            raise ValueError("Invalid normalization state fields")
        if any(state[k] != v for k, v in self.specification().items()):
            raise ValueError("Incompatible normalization specification")
        count = state["count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid normalization count")
        arrays = []
        for name in ("mean", "m2"):
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise ValueError("Statistics must be float64 tensors")
            array = value.detach().cpu().numpy().copy()
            if array.shape != (self.obs_dim,) or not np.isfinite(array).all():
                raise ValueError("Invalid normalization statistics")
            arrays.append(array)
        mean, m2 = arrays
        if (m2 < 0).any() or (count < 2 and np.any(m2)) or (count == 0 and np.any(mean)):
            raise ValueError("Inconsistent normalization moments")
        # Validate cache representability before changing this instance.
        if not np.isfinite(mean.astype(np.float32)).all():
            raise ValueError("Normalization mean exceeds float32 range")
        self.count, self.mean, self.m2 = count, mean, m2
        self._refresh()

    def summary(self) -> dict:
        return {**self.specification(), "count": self.count, "mean": self.mean.tolist(),
                "variance": (self.m2 / self.count if self.count else np.ones(self.obs_dim)).tolist()}
