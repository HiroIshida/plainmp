from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np


class ExtensionResult(Enum):
    REACHED = 0
    ADVANCED = 1
    TRAPPED = 2


@dataclass
class ManifoldRRTConfig:
    n_max_call: int = 2000
    motion_step_shring_rate: float = 0.5


@dataclass
class Node:
    q: np.ndarray
    node_parent: Optional["Node"] = None


class TerminationException(Exception):
    ...


class InvalidStartPosition(Exception):
    ...


class ManifoldRRT(ABC):
    is_reached: Optional[Callable[[np.ndarray], bool]]
    b_min: np.ndarray
    b_max: np.ndarray
    motion_step_box: np.ndarray
    nodes: List[Node]
    f_project: Callable[[np.ndarray, bool], Optional[np.ndarray]]
    f_is_valid: Callable[[np.ndarray], bool]
    f_goal_project: Optional[Callable[[np.ndarray], Optional[np.ndarray]]]
    termination_hook: Optional[Callable[[], None]]
    config: ManifoldRRTConfig
    n_extension_trial: int

    def __init__(
        self,
        start: np.ndarray,
        f_goal_project: Optional[Callable[[np.ndarray], Optional[np.ndarray]]],
        b_min: np.ndarray,
        b_max: np.ndarray,
        motion_step_box: np.ndarray,
        f_project: Callable[[np.ndarray, bool], Optional[np.ndarray]],
        f_is_valid: Callable[[np.ndarray], bool],
        termination_hook: Optional[Callable[[], None]] = None,
        config: ManifoldRRTConfig = ManifoldRRTConfig(),
    ):

        if f_goal_project is not None:
            assert callable(f_goal_project)
        self.f_goal_project = f_goal_project

        if termination_hook is None:

            def termination_hook():
                n_total = self.n_extension_trial
                if n_total > config.n_max_call:
                    raise TerminationException

        self.termination_hook = termination_hook

        self.b_min = b_min
        self.b_max = b_max
        self.nodes = [Node(start, None)]
        self.motion_step_box = motion_step_box
        self.f_project = f_project
        self.f_is_valid = f_is_valid
        self.termination_hook = termination_hook
        self.config = config
        self.n_extension_trial = 0

    @property
    def start_node(self) -> Optional[Node]:
        return self.nodes[0]

    @property
    def dof(self) -> int:
        return len(self.b_min)

    def sample(self) -> np.ndarray:
        q = np.random.rand(self.dof) * (self.b_max - self.b_min) + self.b_min
        return q

    def find_nearest_node(self, q: np.ndarray) -> Node:
        min_idx = np.argmin([np.linalg.norm(q - n.q) for n in self.nodes])
        return self.nodes[min_idx]

    def step(self) -> None:
        q_rand = self.sample()
        self.extend(q_rand)

    def extend(
        self, q_rand: np.ndarray, node_nearest: Optional[Node] = None, collision_aware: bool = False
    ) -> ExtensionResult:

        if self.termination_hook is not None:
            self.termination_hook()

        self.n_extension_trial += 1

        if node_nearest is None:
            node_nearest = self.find_nearest_node(q_rand)
        diff_ambient = q_rand - node_nearest.q

        if np.all(np.abs(diff_ambient) < self.motion_step_box):
            return ExtensionResult.REACHED

        shrink_motion_box = self.motion_step_box * self.config.motion_step_shring_rate
        diff_clamped = np.maximum(np.minimum(diff_ambient, shrink_motion_box), -shrink_motion_box)

        q_here = node_nearest.q + diff_clamped
        for _ in range(5):
            # check if projection successful
            q_new = self.f_project(q_here, collision_aware)
            if q_new is None:
                return ExtensionResult.TRAPPED

            # check if q_new is inside configuration box space
            if np.any(q_new < self.b_min):
                return ExtensionResult.TRAPPED
            if np.any(q_new > self.b_max):
                return ExtensionResult.TRAPPED

            # check if motion step constraint is satisfied
            diff_actual = q_new - node_nearest.q

            if np.linalg.norm(diff_actual) < 1e-6:
                return ExtensionResult.TRAPPED

            if not np.all(np.abs(diff_actual) < self.motion_step_box):
                return ExtensionResult.TRAPPED

            # check if inequality constraint is satisfied
            if not self.f_is_valid(q_new):
                return ExtensionResult.TRAPPED
            q_here = q_new

            new_node = Node(q_here, node_nearest)
            self.nodes.append(new_node)
        return ExtensionResult.ADVANCED

    def connect(self, q_target: np.ndarray) -> Optional[Node]:
        # reutrn connceted Node if connected. Otherwise return None

        self.find_nearest_node(q_target)
        result = self.extend(q_target, None, collision_aware=True)
        if result != ExtensionResult.ADVANCED:
            return None

        dist_pre = np.linalg.norm(q_target - self.nodes[-1].q)
        # because last result is advanced
        while True:
            result = self.extend(q_target, self.nodes[-1], collision_aware=True)
            if result == ExtensionResult.TRAPPED:
                return None
            if result == ExtensionResult.REACHED:
                return self.nodes[-1]
            dist = np.linalg.norm(q_target - self.nodes[-1].q)
            is_deviated = dist > dist_pre
            if is_deviated:
                return None
            dist_pre = dist

    def solve(self) -> bool:
        assert self.start_node is not None
        if not self.f_is_valid(self.start_node.q):
            raise InvalidStartPosition

        assert self.f_goal_project is not None
        try:
            while True:
                q_rand = self.sample()
                res = self.extend(q_rand)
                if res == ExtensionResult.ADVANCED:
                    q_project = self.f_goal_project(self.nodes[-1].q)
                    if q_project is not None:
                        if self.f_is_valid(q_project):
                            result_connect = self.connect(q_project)
                            if result_connect is not None:
                                new_node = Node(q_project, self.nodes[-1])
                                self.nodes.append(new_node)
                                return True
        except TerminationException:
            return False
        return False

    def get_solution(self) -> np.ndarray:
        node = self.nodes[-1]
        q_seq = [node.q]

        while True:
            node = node.node_parent  # type: ignore
            if node is None:
                break
            q_seq.append(node.q)
        q_seq.reverse()
        return np.array(q_seq)

    def visualize(self, fax):
        assert self.dof in [3]
        if fax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = fax
        Q = np.array([n.q for n in self.nodes])

        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=2)
        for n in self.nodes:
            if n.node_parent is None:
                continue
            q = n.q
            q_parent = n.node_parent.q
            ax.plot(
                [q_parent[0], q[0]], [q_parent[1], q[1]], [q_parent[2], q[2]], c="red", linewidth=1
            )


class ManifoldRRTConnect:
    """
    Class for planning a path between two configurations using two ManifoldRRT trees.
    (generated by chat-gpt)

    Parameters
    ----------
    q_start : np.ndarray
        The starting configuration.
    q_goal : np.ndarray
        The goal configuration.
    b_min : np.ndarray
        The minimum values of the configuration space bounds.
    b_max : np.ndarray
        The maximum values of the configuration space bounds.
    motion_step_box : np.ndarray
        The size of the motion step box for the RRT algorithm.
    f_project : Callable[[np.ndarray], Optional[np.ndarray]]
        A function that projects a given configuration onto the manifold.
        return None if projection failed
    f_is_valid : Callable[[np.ndarray], bool]
        A function that checks if a given configuration is valid.
    config : Config, optional
        The configuration object for the RRT algorithm, by default Config().

    Examples
    --------
    Define a sample configuration space and motion model:
    """

    rrt_start: ManifoldRRT
    rrt_goal: ManifoldRRT
    connection: Optional[Tuple[Node, Node]] = None

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        b_min: np.ndarray,
        b_max: np.ndarray,
        motion_step_box: np.ndarray,
        f_project: Callable[[np.ndarray, bool], Optional[np.ndarray]],
        f_is_valid: Callable[[np.ndarray], bool],
        config: ManifoldRRTConfig = ManifoldRRTConfig(),
    ):
        self.rrt_start = ManifoldRRT(
            q_start, None, b_min, b_max, motion_step_box, f_project, f_is_valid, None, config
        )
        self.rrt_goal = ManifoldRRT(
            q_goal, None, b_min, b_max, motion_step_box, f_project, f_is_valid, None, config
        )

        def termination_hook():
            n_total = self.n_extension_trial
            if n_total > config.n_max_call:
                raise TerminationException

        self.rrt_start.termination_hook = termination_hook
        self.rrt_goal.termination_hook = termination_hook

        self.connection = None

    @property
    def n_extension_trial(self) -> int:
        n_total = self.rrt_start.n_extension_trial + self.rrt_goal.n_extension_trial
        return n_total

    def solve(self) -> bool:
        try:
            extend_start_tree = True
            while True:
                if extend_start_tree:
                    rrt_a = self.rrt_start
                    rrt_b = self.rrt_goal
                    extend_start_tree = False
                else:
                    rrt_a = self.rrt_goal
                    rrt_b = self.rrt_start
                    extend_start_tree = True
                q_rand = rrt_a.sample()
                res = rrt_a.extend(q_rand)
                if res == ExtensionResult.ADVANCED:
                    node_target = rrt_a.nodes[-1]
                    node = rrt_b.connect(node_target.q)
                    if node is not None:
                        if extend_start_tree:
                            self.connection = (node, node_target)
                        else:
                            self.connection = (node_target, node)
                        return True
        except TerminationException:
            return False
        return False

    def get_solution(self) -> np.ndarray:
        # TODO: reuse ManifoldRRT
        assert self.connection is not None
        node = self.connection[0]
        q_seq_start = [node.q]
        while True:
            node = node.node_parent  # type: ignore
            if node is None:
                break
            q_seq_start.append(node.q)

        node = self.connection[1]
        q_seq_goal = [node.q]
        while True:
            node = node.node_parent
            if node is None:
                break
            q_seq_goal.append(node.q)

        q_seq = list(reversed(q_seq_start)) + q_seq_goal
        return np.array(q_seq)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def project(q: np.ndarray, hoge) -> Optional[np.ndarray]:
        return q / np.linalg.norm(q)

    def is_valid(q: np.ndarray) -> bool:
        if abs(q[0]) > 0.2:
            return True
        if abs(q[1]) < 0.2:
            return True
        return False

    def f_goal_project(q: np.ndarray) -> Optional[np.ndarray]:
        goal = np.array([+1, 0, 0])
        return goal

    start = np.array([-1, 0, 0])
    b_min = -np.ones(3) * 1.5
    b_max = +np.ones(3) * 1.5
    motion_step_box = np.ones(3) * 0.2
    conf = ManifoldRRTConfig(1000)
    rrt = ManifoldRRT(
        start, f_goal_project, b_min, b_max, motion_step_box, project, is_valid, config=conf
    )
    import time

    ts = time.time()
    res = rrt.solve()
    print(time.time() - ts)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    rrt.visualize((fig, ax))
    Q = rrt.get_solution()
    ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], c="k", linewidth=3)
    plt.show()
