import numpy as np
import faiss
import collections
import operator
import typing

class DistanceFunc(typing.Protocol):
    def __call__(self, a: typing.Any, b: typing.Any, **kwargs) -> float:
        ...

class FunctionWrapper:
    def __init__(self, distance_function: DistanceFunc):
        self.distance_function = distance_function

    def __call__(self, a, b):
        # Access x, which is stored in a tuple (x, y)
        return self.distance_function(a[0], b[0])

class NearestNeighbors:
    def __init__(
        self,
        window_size: int,
        min_distance_keep: float,
        distance_func: typing.Union[DistanceFunc, FunctionWrapper],
        k: int = 1,
    ):
        self.window_size = window_size
        self.min_distance_keep = min_distance_keep
        self.distance_func = distance_func
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self.k = k
        self.index = None

    def append(self, item: typing.Any, extra: typing.Optional[typing.Any] = None):
        self.window.append((item, *(extra or [])))
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(item[0]))
        if self.index.ntotal >= self.window_size:
            # Remove the oldest item from the Faiss index
            self.index.remove_ids(np.array([self.index.ntotal - self.window_size], dtype=np.int64))
        self.index.add(np.array(list(item[0].values()), dtype=np.float32).reshape(-1, len(item[0])))


    def update(
        self,
        item: typing.Any,
        n_neighbors: int = 1,
        extra: typing.Optional[typing.Any] = None,
    ):
        if self.min_distance_keep == 0:
            self.append(item, extra=extra)
            return True

        nearest = self.find_nearest(item, n_neighbors)

        if not nearest or nearest[0][-1] < self.min_distance_keep:
            self.append(item, extra=extra)
            return True
        return False

    def find_nearest(self, item: typing.Any, n_neighbors: int = 1):
        if self.index is None:
            return None

        distances, indices = self.index.search(np.array(list(item[0].values()), dtype=np.float32).reshape(-1, len(item[0])), n_neighbors)
        points = [(*self.window[i], d) for i, d in zip(indices[0], distances[0])]
        return sorted(points, key=operator.itemgetter(-1))[:n_neighbors]
