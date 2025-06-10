import copy
import json
from typing import Callable, List, Optional, Tuple, overload

import numpy as np


class InvalidSamplePointError(Exception):
    pass


class EuclideanMetric:
    def __call__(self, p0: np.ndarray, p1: np.ndarray) -> float:
        return float(np.linalg.norm(p0 - p1))


class Trajectory:
    """Resamplable trajectory for robot motion.

    This class represents a robot trajectory as a sequence of waypoints with
    support for resampling, length computation, and trajectory manipulation.
    It supports custom distance metrics for different robot types.

    Parameters
    ----------
    points : List[np.ndarray]
        Sequence of waypoints (joint configurations).
    metric : Callable[[np.ndarray, np.ndarray], float], default=EuclideanMetric()
        Distance metric for trajectory calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from plainmp.trajectory import Trajectory
    >>>
    >>> # Create trajectory from waypoints
    >>> waypoints = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])]
    >>> traj = Trajectory(waypoints)
    >>>
    >>> # Resample to 10 waypoints
    >>> resampled = traj.resample(10)
    >>>
    >>> # Get trajectory length
    >>> length = traj.get_length()
    >>>
    >>> # Sample point at specific distance
    >>> mid_point = traj.sample_point(length / 2)

    Notes
    -----
    If using a non-Euclidean metric, get_length(), sample_point(), and resample()
    are approximations because geodesic distances are not linearly additive.
    For accurate results with non-Euclidean metrics, prepare trajectories with
    many waypoints before resampling.
    """

    _points: List[np.ndarray]
    metric: Callable[[np.ndarray, np.ndarray], float]
    _dist_cache: List[Optional[float]]

    def __init__(
        self,
        points: List[np.ndarray],
        metric: Callable[[np.ndarray, np.ndarray], float] = EuclideanMetric(),
    ):
        self._points = points
        self.metric = metric
        self._dist_cache = [0] + [None] * (len(points) - 1)

    def dumps(self) -> str:
        points = [point.tolist() for point in self._points]
        assert isinstance(self.metric, EuclideanMetric)
        return json.dumps({"points": points})

    @classmethod
    def loads(cls, s: str) -> "Trajectory":
        data = json.loads(s)
        points = [np.array(point) for point in data["points"]]
        [0] + [None] * (len(points) - 1)
        return cls(points, EuclideanMetric())

    def get_metric_changed(self, metric: Callable[[np.ndarray, np.ndarray], float]) -> "Trajectory":
        return Trajectory(self._points, metric)

    def get_length_from_start(self, index: int) -> float:
        if not 0 <= index < len(self._points):
            raise IndexError("Index out of bounds")

        cached_value = self._dist_cache[index]
        if cached_value is not None:
            return cached_value

        start_index = next(i for i, x in enumerate(self._dist_cache) if x is None)
        for i in range(start_index, len(self._points)):
            if i == 0:
                self._dist_cache[i] = 0
            else:
                self._dist_cache[i] = self._dist_cache[i - 1] + self.metric(  # type: ignore[operator]
                    self._points[i], self._points[i - 1]
                )
        return self._dist_cache[index]  # type: ignore[return-value]

    def get_length(self) -> float:
        # NOTE: see NOTE in class docstring if metric is non-euclidean
        return self.get_length_from_start(len(self) - 1)

    def _sample_point(self, dist_from_start: float) -> np.ndarray:
        # NOTE: see NOTE in class docstring if metric is non-euclidean

        L = self.get_length()
        if dist_from_start > L + 1e-6:
            raise InvalidSamplePointError("exceed total length")

        dist_from_start = min(dist_from_start, L)
        edge_dist_sum = 0.0
        for i in range(len(self) - 1):
            edge_dist_sum += self.metric(self._points[i + 1], self._points[i])
            if dist_from_start <= edge_dist_sum:
                diff = edge_dist_sum - dist_from_start
                vec_to_prev = self._points[i] - self._points[i + 1]
                vec_to_prev_unit = vec_to_prev / self.metric(self._points[i], self._points[i + 1])
                point_new = self._points[i + 1] + vec_to_prev_unit * diff
                return point_new
        raise InvalidSamplePointError()

    def sample_point(self, dist_from_start: float) -> np.ndarray:
        L = self.get_length()
        if dist_from_start > L + 1e-6:
            raise InvalidSamplePointError("exceed total length")

        dist_from_start = min(dist_from_start, L)

        # Iterate over segments to find the correct one
        for i in range(1, len(self)):
            segment_end_length = self.get_length_from_start(i)
            if dist_from_start <= segment_end_length:
                segment_start_length = self.get_length_from_start(i - 1)
                dist_on_edge = dist_from_start - segment_start_length
                edge_length = segment_end_length - segment_start_length
                interpolation_ratio = dist_on_edge / edge_length
                point_new = self._points[i - 1] + interpolation_ratio * (
                    self._points[i] - self._points[i - 1]
                )
                return point_new

        raise InvalidSamplePointError()

    def resample(self, n_waypoint: int) -> "Trajectory":
        """Resample trajectory to specified number of waypoints.

        Parameters
        ----------
        n_waypoint : int
            Number of waypoints in the resampled trajectory.

        Returns
        -------
        Trajectory
            New trajectory with evenly spaced waypoints.

        Examples
        --------
        >>> traj = Trajectory([q1, q2, q3])
        >>> smooth_traj = traj.resample(50)  # 50 evenly spaced waypoints
        """
        # NOTE: see NOTE in class docstring if metric is non-euclidean
        L = self.get_length()
        point_new_list = []
        partial_length = L / (n_waypoint - 1)
        for i in range(n_waypoint):
            dist_from_start = partial_length * i
            point_new = self.sample_point(dist_from_start)
            point_new_list.append(point_new)
        return Trajectory(point_new_list, self.metric)

    def get_duplicate_removed(self) -> "Trajectory":
        point_new_list: List[np.ndarray] = []
        eps = 1e-4
        for point in self._points:
            if len(point_new_list) == 0:
                point_new_list.append(point)
            else:
                diff_from_prev = self._points[-1] - point
                if np.all(np.abs(diff_from_prev) > eps):
                    point_new_list.append(point)
        return Trajectory(point_new_list, self.metric)

    def numpy(self):
        return np.array(self._points)

    def visualize(self, fax: Tuple, *args, **kwargs) -> None:
        fig, ax = fax
        arr = self.numpy()
        ax.plot(arr[:, 0], arr[:, 1], *args, **kwargs)

    @classmethod
    def from_two_points(cls, start: np.ndarray, goal: np.ndarray, n_waypoint) -> "Trajectory":
        diff = goal - start
        points = [start + diff / (n_waypoint - 1) * i for i in range(n_waypoint)]
        return cls(points, EuclideanMetric())  # only support EuclideanMetric

    @overload
    def __getitem__(self, indices: List[int]) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, indices: slice) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        pass

    def __getitem__(self, indices_like):
        points = self._points
        return points[indices_like]  # type: ignore

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return self._points.__iter__()

    def __add__(self, other: "Trajectory") -> "Trajectory":
        assert self.metric is other.metric
        diff_contact = np.linalg.norm(self._points[-1] - other._points[0])
        assert diff_contact < 1e-6
        points = copy.deepcopy(self._points) + copy.deepcopy(other._points[1:])
        return Trajectory(points, self.metric)
