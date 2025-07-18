"""Strategy wrappers for different k-NN backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import sklearn.neighbors

from cellmapper.check import check_deps


class _KNNBackend(ABC):
    """Abstract interface for a k-NN search backend."""

    @abstractmethod
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ): ...

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Build any index or data structure on `data`."""
        ...

    @abstractmethod
    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:  # (distances, indices)
        """Return (distances, indices) of the k nearest neighbors of `points`."""
        ...


class _SklearnBackend(_KNNBackend):
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ):
        self._nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)

    def fit(self, data: np.ndarray) -> None:
        self._nn.fit(data)

    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self._nn.kneighbors(points, n_neighbors=k)
        return distances, indices


class _FaissCpuBackend(_KNNBackend):
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ):
        check_deps("faiss-cpu")
        import faiss

        self.faiss = faiss
        self._index = None

    def fit(self, data: np.ndarray) -> None:
        dims = data.shape[1]
        index = self.faiss.IndexFlatL2(dims)
        # Ensure data is float32 and C-contiguous
        data_f32 = np.ascontiguousarray(data.astype(np.float32))
        index.add(data_f32)
        self._index = index

    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Ensure points are float32 and C-contiguous
        points_f32 = np.ascontiguousarray(points.astype(np.float32))
        distances, indices = self._index.search(points_f32, k)
        return distances, indices


class _FaissGpuBackend(_KNNBackend):
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ):
        check_deps("faiss-gpu")
        import faiss

        self.faiss = faiss
        self.res = faiss.StandardGpuResources()
        self._index = None

    def fit(self, data: np.ndarray) -> None:
        dims = data.shape[1]
        flat = self.faiss.IndexFlatL2(dims)
        gpu_index = self.faiss.index_cpu_to_gpu(self.res, 0, flat)
        # Ensure data is float32 and C-contiguous
        data_f32 = np.ascontiguousarray(data.astype(np.float32))
        gpu_index.add(data_f32)
        self._index = gpu_index

    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Ensure points are float32 and C-contiguous
        points_f32 = np.ascontiguousarray(points.astype(np.float32))
        distances, indices = self._index.search(points_f32, k)
        return distances, indices


class _RapidsBackend(_KNNBackend):
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ):
        check_deps("cuml")
        import cuml as cm

        check_deps("cupy")
        import cupy as cp

        self.cm = cm
        self.cp = cp
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwargs = kwargs
        self._nn = None

    def fit(self, data: np.ndarray) -> None:
        data_gpu = self.cp.asarray(data)
        self._nn = self.cm.neighbors.NearestNeighbors(
            n_neighbors=self.n_neighbors,
            output_type="numpy",
            metric=self.metric,
            **self.kwargs,
        ).fit(data_gpu)

    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        points_gpu = self.cp.asarray(points)
        distances, indices = self._nn.kneighbors(points_gpu)
        return distances, indices


class _PyNNDescentBackend(_KNNBackend):
    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        random_state: int = 0,
        **kwargs: Any,
    ):
        check_deps("pynndescent")
        from pynndescent import NNDescent

        self.NNDescent = NNDescent
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs
        self._index = None

    def fit(self, data: np.ndarray) -> None:
        params = self.kwargs.copy()
        if "n_jobs" not in params:
            params["n_jobs"] = -1
        if "n_trees" not in params:
            params["n_trees"] = min(64, 5 + round(data.shape[0] ** 0.5 / 20.0))
        if "n_iters" not in params:
            params["n_iters"] = max(5, round(np.log2(data.shape[0])))
        self._index = self.NNDescent(data, metric=self.metric, random_state=self.random_state, **params)

    def query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # NNDescent.query returns (indices, distances)
        indices, distances = self._index.query(points, k=k)
        return distances, indices


_BACKENDS = {
    "sklearn": _SklearnBackend,
    "faiss-cpu": _FaissCpuBackend,
    "faiss-gpu": _FaissGpuBackend,
    "rapids": _RapidsBackend,
    "pynndescent": _PyNNDescentBackend,
}


def get_backend(knn_method: str, n_neighbors: int, metric: str, random_state: int = 0, **kwargs: Any) -> _KNNBackend:
    """Factory to get a configured KNN backend."""
    try:
        backend_cls = _BACKENDS[knn_method]
    except KeyError:
        raise ValueError(f"Unknown method: {knn_method}. Supported methods: {list(_BACKENDS)}") from KeyError
    return backend_cls(n_neighbors=n_neighbors, metric=metric, random_state=random_state, **kwargs)
